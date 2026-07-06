# План: измерение и декомпозиция времени спекулятивного декодирования (vLLM V1)

> Статус: исследование завершено (2 раунда), код не написан. Документ — спецификация реализации.
> Дата: 2026-06-25. Репозиторий: clone upstream `vllm-project/vllm` (V1-движок).
> Цель воспроизводит измерения статьи «When Speculative Decoding Meets Mixture-of-Experts:
> Target Verification Cost as a Bottleneck» (Qwen3-30B-A3B + EAGLE-3, A100, SpecBench, greedy T=0).

## 0. Что измеряем (из статьи)

**Тайминги по раунду:** `T_SD` (полный шаг), `T_D,r` (draft), `T_T,r = T_T^(B)(K)` (target-верификация
K+bonus), `T_reject,r` (rejection+постобработка), `T_T^(B)(0)` (авторегр. база), `T_D^(B)(1)` (1 draft-токен).

**Диагностики:** Accepted length `𝔼[A]=S/R`, Target Efficiency `η(K)=T_T(0)/T_T(K)`,
draft-target ratio `R_DT=K·T_D/T_T(0)`, Speedup.

**MoE-сигнал (ядро):** `Ū_r` — layer-averaged distinct-expert count по верификационному микробатчу
(`Ū_r = (1/|L_MoE|) Σ_ℓ |U_{ℓ,r}|`, где `U_{ℓ,r}` — union TopK-экспертов по позициям слоя ℓ);
линейная модель `T_T,r ≈ α·Ū_r + β`, пер-k коэффициенты `α_k, β_k` (Table 2).

**Главная декомпозиция (§4.5, Fig.4, Table 1):**
`T_SD = T*_id + Δ_partition + Δ_rejection`, где
`Δ_partition = T_id − T*_id` (штраф геометрии раундов),
`Δ_rejection = T_SD − T_id` (штраф качества draft-предсказаний),
`T_id` — ideal-draft rerun (та же policy, 100% accept), `T*_id` — Bellman-DP oracle.

## 1. Карта: величина статьи → vLLM

| Величина | Где в vLLM | Статус |
|---|---|---|
| `T_SD` | сумма стадий / wall-clock шага | 🔶 Tier 1 |
| `T_D,r` | `draft_total_ms` + per-pos (`llm_base_proposer` K-цикл) | 🔶 Tier 1 |
| `T_T,r = T_T(K)` | `target_forward_ms` (`_model_forward` @ `gpu_model_runner.py:4332`) | 🔶 Tier 1 |
| `T_reject,r` | `verify_ms` (`rejection_sampler` @ `:3605`) | 🔶 Tier 1 |
| `T_T(0)` | `target_forward_ms` на non-spec шагах (тэг по числу позиций) | 🔶 Tier 1 |
| `T_D(1)` | `draft_forward_ms_per_pos[0]` | 🔶 Tier 1 |
| `𝔼[A]` | `num_accepted/num_drafts` (`metrics.py:114`) | ✅ есть |
| per-pos acceptance | `num_accepted_tokens_per_pos` | ✅ есть |
| `η(K)`, `R_DT` | производные | 📊 расчёт |
| **`Ū_r` / `U_{ℓ,r}` / `𝓔_{ℓ,b,t}`** | `RoutedExpertsCapturer.routing_data` + distinct-count | ✅ данные / 🔶 +редукция (Tier 2) |
| `α_k, β_k` (Table 2) | фит из `(Ū_r, T_T(K))` или калибровочный свип | 📊 Tier 3 |
| per-round `(K, a_r)` | `scheduler.py:1553-1556` | ✅ доступно (нужен трейс) |
| **`T_SD = T*_id+Δ_part+Δ_rej`** | оффлайн: replay + ideal-rerun + Bellman-DP | 📊 Tier 3 |
| greedy `T=0` | `SamplingParams(temperature=0, seed)` (`gpu_input_batch.py:1091`) | ✅ есть |

## 2. Архитектура (3 уровня)

```
TIER 1 — online stage timing (CUDA events, double-buffer)
   T_SD, T_D, T_T(K), T_reject, T_T(0), T_D(1)  + тэг target числом верифиц. позиций
TIER 2 — online MoE-сигнал
   Ū_r из RoutedExpertsCapturer.routing_data (distinct per layer → среднее)
TIER 3 — offline-декомпозиция (потребляет per-round трейс)
   трейс → cost-fit(α_k,β_k) → ideal-rerun(T_id) → Bellman-DP(T*_id)
        → Δ_partition, Δ_rejection → Table 1, Fig 2–5
```

Tier 1–2 — патч vLLM (online, горячий путь). Tier 3 — отдельный Python-пакет (оффлайн).

---

## 3. TIER 1 — online stage timing

### 3.1 Главная техническая проблема
CUDA async + CUDA Graphs: `time.perf_counter()` вокруг `self.model(...)` меряет только запуск.
Нужны `torch.cuda.Event(enable_timing=True)`.

### 3.2 Вердикты осуществимости (round-1 research)
- **Per-position draft — ВОЗМОЖНО:** proposer всегда `PIECEWISE` (`llm_base_proposer.py:401-410`),
  каждый из K draft-форвардов — отдельный replay; пара событий обрамляет каждый. Guard: при FULL-графе
  draft → откат на coarse.
- **Async-output — double-buffer:** события пишем в шаге N, `elapsed_time` читаем в начале N+1
  (compute-стрим серийный → готово без новых sync). Mode-independent (SYNC/ASYNC). НИКОГДА не
  синхронизировать внутри K-цикла.
- **Слияние worker→метрики:** транспорт через новое поле `ModelRunnerOutput.spec_decode_timing`
  (зеркало `cudagraph_stats` @ `outputs.py:270`); scheduler читает на `:1475`, вливает в
  `SpecDecodingStats` один раз/шаг перед `make_stats` (`:1791`). Тайминг — per-step.

### 3.3 Точки и слои
| Стадия | Точка | Покрытие |
|---|---|---|
| `target_forward_ms` | `_model_forward` @ `gpu_model_runner.py:4332` | все |
| `verify_ms` | `rejection_sampler` @ `:3605` | spec |
| `sample_ms` | `sampler` @ `:3593` | все |
| `draft_total_ms` (coarse) | `self.drafter.propose()` @ `:5122` | все proposer'ы (ngram/suffix тоже) |
| `draft_forward_ms_per_pos[K]` (fine) | `self.model()` @ `llm_base_proposer.py:540,683` | model-based |

### 3.4 Принципы
1. Никаких `synchronize()` в горячем пути; дренаж — double-buffer.
2. Пул событий (ping-pong глубины 2), не аллокация на шаг.
3. Тайминг per-step.
4. Массивы per-pos — на `max num_speculative_tokens`.
5. **Тэг target числом верифиц. позиций** (`Σ(K_i+1)` из `SpecDecodeMetadata.cu_num_sampled_tokens`)
   — нужно для `T_T(k)` cost-model и `T_T(0)`.

---

## 4. TIER 2 — Ū_r (distinct-expert)

**Реюз:** `RoutedExpertsCapturer` (`vllm/model_executor/layers/fused_moe/routed_experts_capturer.py`)
уже пишет per-layer `topk_ids` в device-буфер `(max_num_batched_tokens, num_layers, num_experts_per_tok)`
int32; CUDA-graph-совместимо (фиксация до графа через callback в `select_experts` @ `moe_runner.py:560`);
экспорт `ModelRunnerOutput.routed_experts`. Гейт — `--enable-return-routed-experts`
(`config/model.py:214`).

**Что добавить — distinct-count редукция per-step:**
для каждого слоя ℓ — число различных экспертов в union'е `topk_ids` по токенам шага (K+1 микробатч),
среднее по слоям → `Ū_r`. Варианты:
- **A (host, простой):** presence-матрица `(num_layers, num_experts)` через scatter из `routing_data`,
  `Ū_r = mean_ℓ presence[ℓ].sum()`.
- **B (реюз EPLB):** если EPLB включён — `expert_load_pass` shape `(num_layers, num_experts)` уже считает
  токены/эксперт; `Ū_r = mean_ℓ (expert_load_pass[ℓ] > 0).sum()`.
- **C (device kernel):** scatter presence на GPU (near-template — EPLB `_eplb_map_and_record` kernel
  в `base_router.py`), редукция per-layer. Дешевле всего на масштабе.

**Транспорт:** в тот же `SpecDecodeTimingStats` / `ModelRunnerOutput.spec_decode_timing`.
Хранить `Ū_r` (float, per-step) + опц. вектор `|U_{ℓ,r}|` по слоям. Для cost-model/oracle в трейс
дополнительно идёт per-position routing (`routing_data`).

---

## 5. TIER 3 — offline-декомпозиция (полная)

### 5.1 Per-round трейс-сток (НОВОЕ; нативного нет)
Хук/`StatLogger`, пишущий per (round_id=`scheduler.current_step`, req_id):
`K` (`scheduler.py:1553`), `a_r` accepted (`:1555`), num верифиц. позиций
(`SpecDecodeMetadata.cu_num_sampled_tokens`), `T_T(K)`, `T_D`, `T_reject`, `Ū_r`,
per-position routing (`routing_data`) для позиций раунда. Формат: JSONL/parquet,
greedy T=0 прогон по SpecBench (480 вопросов).
> Точки перехвата сырых per-request значений: `scheduler.py:1547-1574` (K, a_r) и
> `gpu_model_runner.py:_calc_spec_decode_metadata` (`:2750`, структура микробатча).

### 5.2 Оффлайн-пакет (новый, напр. `tools/spec_decode_decomp/`)
1. **cost-fit** `α_k,β_k`: из `(Ū, T_T)` по раундам, бины по k → Table 2; `T_T(0)` из non-spec шагов.
2. **ideal-rerun → `T_id`**: та же online-policy на том же greedy target-выходе, rejected suffix убран
   (`r_i = ∅`), стоимости раундов — через cost-model.
3. **Bellman-DP → `T*_id`**: `OPT(i)=min_{k≤K_i}[C(i,k)+OPT(i+k+1)]`,
   `C(i,k)=k·T_D(1)+T_T(k, 𝓔_{i:i+k})`, `K_i=min(L,N−i)`, `L=K`. `O(N·L)`.
   `𝓔_{i:i+k}` (union экспертов на интервале) считается из per-position routing трейса.
4. **Декомпозиция:** `Δ_partition=T_id−T*_id`, `Δ_rejection=T_SD−T_id`.
5. **Выводы:** Table 1 (по K: `𝔼[A]`, Speedup, `Δ_rejection/T_SD`, `Δ_partition/T_SD`, `T*_id/T_SD`),
   Fig 4 (стек), Fig 2/3 (`T_T` vs `Ū_r`, scatter), Fig 5 (cross-domain).

### 5.3 Детерминизм
greedy `temperature=0` + фиксированный `seed` (`SamplingParams`); проверка `all_greedy`
(`gpu_input_batch.py:1091`). Нужен для воспроизводимого target-выхода в ideal-rerun.

---

## 6. Формат вывода — нативный vLLM `get_metrics()`

`vllm/v1/metrics/reader.py::get_metrics_snapshot()` (= `llm.get_metrics()`) возвращает `list[Metric]`:
`Counter` / `Gauge` / `Vector` (массив!) / `Histogram` (`name`+`labels`+значения).

- **Per-position / per-K массивы** моделируем паттерном `num_accepted_tokens_per_pos`:
  Counter с label `position`/`k` → reader сворачивает в **`Vector`** (`_digest_num_accepted_by_pos_samples`).
- Tier 1–2 метрики регистрируем в `SpecDecodingProm` (`metrics.py:177`) → автоматически в `get_metrics()`.
- Tier 3 результаты (Δ-ы, T*_id… по K) → `Gauge` (скаляры) + `Vector` (массивы по K).
- **Решение:** `Vector.values: list[int]` — тайминг float. Хранить целые **микросекунды**, либо
  расширить reader новым float-вектором. (По умолчанию — микросекунды-int, минимально инвазивно.)

Имена (предложение): `vllm:spec_decode_target_forward_seconds`, `..._verify_seconds`,
`..._draft_total_seconds`, `..._draft_forward_seconds` (Vector по position), `..._distinct_experts_avg`
(Gauge), `..._distinct_experts_per_layer` (Vector), `..._rejection_penalty_ratio` / `..._partition_penalty_ratio`
(Vector по K, Tier 3).

---

## 7. Touch-list

| Файл | Изменение |
|---|---|
| `vllm/config/observability.py:56` | + флаг `spec_decode_timing: bool = False` |
| `vllm/v1/spec_decode/timing.py` *(new)* | `SpecDecodeTimer` (пул событий, `time_stage`, `drain_previous`) + `SpecDecodeTimingStats` (timing + `Ū_r`) |
| `vllm/v1/spec_decode/metrics.py` | timing/`Ū_r`-поля + `observe_timing()` в `SpecDecodingStats`; агрегат+лог в `SpecDecodingLogging`; counter'ы/`Gauge`/`Vector` в `SpecDecodingProm` |
| `vllm/v1/outputs.py:234,270` | + `ModelRunnerOutput.spec_decode_timing` |
| `vllm/v1/worker/gpu_model_runner.py` | `SpecDecodeTimer`; обёртки `:4332/:3605/:3593/:5122`; `Ū_r` из `routing_data`; drain+populate `:4621`; тэг target позициями |
| `vllm/v1/spec_decode/llm_base_proposer.py` | per-position `time_stage("draft", pos=i)` @ `:540,683` |
| `vllm/v1/core/sched/scheduler.py:1475,1791` | прочитать `spec_decode_timing`, влить в `SpecDecodingStats` 1×/шаг |
| `vllm/v1/spec_decode/trace.py` *(new)* | per-round трейс-сток (`StatLogger`/хук) → JSONL/parquet |
| `tools/spec_decode_decomp/` *(new)* | cost-fit + ideal-rerun + Bellman-DP + Table 1/Figures; вывод как `list[Metric]` |
| `tests/v1/spec_decode/` | unit: `observe_timing`, distinct-count, Bellman-DP; интеграц.: флаг on/off (off → 0 оверхед) |

## 8. Фазы

1. **Каркас (без горячего пути):** флаг + `SpecDecodeTimer` + `SpecDecodeTimingStats` + расширение
   `SpecDecodingStats`/`Prom`/`Logging`. Метрики пустые, ничего не ломаем.
2. **Tier 1 инструментирование:** target/verify/sample/draft (coarse) + тэг позициями.
3. **Tier 1 fine:** per-position draft в `llm_base_proposer`.
4. **Транспорт + слияние + экспорт** (`get_metrics()` Vector/Gauge).
5. **Tier 2:** `Ū_r` из `RoutedExpertsCapturer` (distinct-count) + экспорт.
6. **Tier 3a:** per-round трейс-сток (greedy T=0).
7. **Tier 3b:** оффлайн-пакет — cost-fit → ideal-rerun → Bellman-DP → Δ-декомпозиция → Table 1/Figures.
8. **Тесты** на каждом уровне.

## 9. Краевые случаи
- **ngram/suffix** — нет draft-форварда → только `draft_total_ms`; per-pos пустой.
- **dynamic spec decode** — переменный K → массивы на `max`; в Bellman `K_i=min(L,N−i)`.
- **diffusion (dLLM)** — отдельные имена (как acceptance) или пропуск.
- **FULL-граф draft** — guard на coarse.
- **EAGLE-3 hidden states** — форварды те же (`:540/683`), per-position применим.
- **monolithic MoE-kernel** — ✅ РИСК СНЯТ (round-4 проверка). Для Qwen3-30B-A3B (BF16, GPU/A100)
  `is_monolithic=False` (monolithic только CPU: `unquantized_fused_moe_method.py:58-62` + коммент `:72-75`)
  → modular-путь → `select_experts` (`moe_runner.py:560`) → `capture_fn(topk_ids)` (`base_router.py:269`)
  срабатывает. Реюз `RoutedExpertsCapturer` для `Ū_r` РАБОТАЕТ. EPLB-fallback тоже есть
  (`supports_eplb=True`).
- **V1 model runner (важно для scope)** — вся инструментация (Tier 1 timing + Tier 2 `Ū_r`) живёт в
  V1 `vllm/v1/worker/gpu_model_runner.py`. Флаг `--enable-return-routed-experts` авто-форсит V1
  (V2-раннер `vllm/v1/worker/gpu/model_runner.py` пока не поддерживает capture — `config/vllm.py:2057`,
  PR #38163; `use_v2_model_runner→False` с warning, не ошибка). НЕ ставить `VLLM_USE_V2_MODEL_RUNNER=1`.
- **CUDA-graph для Ū_r** — совместимо (фиксация до графа); per-step Ū_r читать в double-buffer.

## 10. Открытые вопросы
Полный список и карта «фаза → блокеры» — в `SPEC_DECODE_OPEN_QUESTIONS.md`.
**Фаза 1 не блокируется** (единственная связь — тип метрики C8, дефолт: микросекунды-int → `Vector`).
Ответы автора нужны к Tier 3: A1 (эмулятор `T_id`), A2 (калибровка `α_k,β_k`), A3 (`T_AR`),
A4 (B=1), C9 (формат трейса), C11 (толерантность валидации).
B5–B7 (границы стадий, EAGLE-3) и D2 (logical/physical id) закрываются чтением кода к Фазам 2–3, 5.

## 11. Связанная память
`~/.claude/projects/-Users-damamatin-Desktop-vllm-evict/memory/vllm-spec-decode-timing.md`
