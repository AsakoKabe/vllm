# Spec-decode metrics — reference

Справочник по метрикам спекулятивного декодирования: **имя · описание · формула · где считается · где экспортируется**. Формулы — в нотации статьи (LaTeX). Ветка `feat/spec-decode-timing`.

## Как включить и прочитать
- Тайминг-метрики (раздел 1) требуют флаг **`--spec-decode-timing`** (форсит V1 model runner); любые метрики — **`disable_log_stats=False`**.
- Чтение: `llm.get_metrics()` (`vllm/v1/metrics/reader.py`) → `list[Metric]` (`Counter`/`Gauge`/`Vector`).
- Единица тайминга в экспорте — целые **микросекунды** (внутри мс через CUDA-events, ×1000, чтобы пережить `int()`-коэрцию reader'а). Для мс делить на 1000.

## 1. Тайминг-метрики стадий (Tier 1)
Кумулятивные `Counter` (µs), меряются CUDA-events (`SpecDecodeTimer`, `vllm/v1/spec_decode/timing.py`), дренаж с лагом в 1 шаг. Значение $=\sum_{\text{steps}}\text{stage\_ms}\cdot 1000$; среднее на шаг (мс) $=\text{counter}_{\mu s}/\text{num\_timed\_steps}/1000$.

| Метрика `vllm:spec_decode_…` | Описание | Нотация | Где считается → экспорт |
|---|---|---|---|
| `…target_forward_microseconds` | Forward target-**модели** по $K{+}1$ верифиц. позициям (attention+MoE/MLP). Самая дорогая стадия; для MoE растёт с $K$ через $\bar U_r$. Без LM-head. | $T_T^{(B)}(K)$ | `gpu_model_runner.py:4359` → `metrics.py:347` |
| `…verify_microseconds` | Rejection sampling: accept/reject драфтов + сэмпл bonus. Дёшево (sampling-kernel). | $T_{\text{reject},r}$ | `gpu_model_runner.py:3626` → `metrics.py:351` |
| `…sample_microseconds` | Обычный сэмплер на non-spec шагах (нет драфтов). В spec-прогоне $\approx 0$. | — | `gpu_model_runner.py:3613` → `metrics.py:354` |
| `…draft_total_microseconds` | Вся `propose()`: input prep + $K$ форвардов draft-модели + сэмпл draft-токенов. $(\text{draft\_total}-\sum_i\text{per\_pos}_i)\approx$ оркестрация. | $T_{D,r}$ | `gpu_model_runner.py:4525` → `metrics.py:356` |
| `…draft_forward_microseconds_per_pos` | Каждый из $K$ форвардов draft-модели отдельно (позиция $i$). `Vector`. | $T_D^{(B)}(1)$ | `llm_base_proposer.py:544,688` → `metrics.py:386` |
| `…num_timed_steps` | Число спек. шагов с влитым таймингом — знаменатель для средних. | $R$ | `scheduler.py:1819` → `metrics.py:360` |
| `…distinct_experts_milli` | $\bar U_r$ ×1000, суммарно по шагам; mean $\bar U_r$ = value/1000/num_timed_steps. Только MoE-таргет; требует `--enable-return-routed-experts`. | $\bar U_r$ | `gpu_model_runner.py:4655` → `metrics.py:397` |
| `…target_forward_microseconds_by_positions` | `target_forward`, суммированный по шагам, верифицировавшим ровно $k$ позиций (индекс = $k$). `Vector`. **Чистый источник $T_T$** (в отличие от скаляра — без prefill). | $\sum T_T$ для $k$ поз. | `metrics.py` observe → `reader.py` |
| `…target_forward_count_by_positions` | Число шагов с ровно $k$ верифиц. позициями (знаменатель для предыдущей). `Vector`. | $\#$ шагов | `metrics.py` observe → `reader.py` |

**`target_forward` ≠ `verify`:** `target_forward` ($T_T(K)$) — прогон target-**модели** (тяжёлый compute); `verify` ($T_{\text{reject}}$) — **алгоритм** accept/reject поверх готовых логитов (дёшево). `sample`/`verify` взаимоисключающие.

**$T_T(0)$, $\eta(K)$, $R_{DT}$ из бинов.** Пусть $k$ — число верифицированных позиций (= $K_{\text{draft}}{+}1$; при $B{=}1$ это $K{+}1$). Тогда $T_T(j)\,[\text{ms}] = \dfrac{\texttt{by\_positions}[j{+}1]}{\texttt{count}[j{+}1]\cdot 1000}$; отсюда $T_T(0)=\text{bin}[1]$, $T_T(K)=\text{bin}[K{+}1]$, $\eta(K)=\dfrac{T_T(0)}{T_T(K)}$, $R_{DT}=\dfrac{K\,T_D}{T_T(0)}$. Скаляр `…target_forward_microseconds` смешивает prefill-форварды ($k{=}0$) — для $T_T(k)$ бери бины. Prefill ($k{=}0$) и мульти-seq $B{>}1$ ($k>K{+}1$) в бины не пишутся. Лог-строка `SpecDecoding T_T by positions …` печатает $T_T(j)$ по бинам + $\eta(K)$ (индекс $j$ — в нотации статьи).

## 2. Формулы из статьи (LaTeX)

**Диагностика скорости.**

$$\mathbb{E}[A] = \frac{1}{R}\sum_{r=1}^{R} A_r = \frac{S}{R}, \qquad \eta_B(K) = \frac{T_T^{(B)}(0)}{T_T^{(B)}(K)}, \qquad R_{DT} = \frac{K\,T_D}{T_T(0)}$$

$$\text{SpeedUp} = \frac{\mathbb{E}[A]}{K\dfrac{T_D}{T_T(0)} + \dfrac{1}{\eta(K)} + \dfrac{T_{\text{reject}}}{T_T(0)}} \approx \frac{\mathbb{E}[A]}{K\dfrac{T_D}{T_T(0)} + \dfrac{1}{\eta(K)}}$$

**MoE-роутинг (сигнал $\bar U_r$).**

$$\mathcal{E}_{\ell,b,t} = \mathrm{TopK}\big(G_\ell(x_{\ell,b,t}),\, k_e\big), \qquad \mathcal{U}_{\ell,r} = \bigcup_{(b,t)\in\mathcal{I}_r} \mathcal{E}_{\ell,b,t}, \qquad \bar{U}_r = \frac{1}{|\mathcal{L}_{\mathrm{MoE}}|}\sum_{\ell\in\mathcal{L}_{\mathrm{MoE}}} |\mathcal{U}_{\ell,r}|$$

$$T_{T,r} \approx \alpha\,\bar{U}_r + \beta$$

**Декомпозиция $T_{SD}$ (Tier 3).**

$$T_{SD}(\pi) = \sum_{i=1}^{|\pi|} C_i, \qquad C_i = (|a_i| + |r_i|)\,T_D(1) + T_T([a_i, r_i])$$

$$OPT(i) = \min_{k\le K_i}\big[\,C(i,k) + OPT(i{+}k{+}1)\,\big], \qquad C(i,k) = k\,T_D(1) + T_T(k,\mathcal{E}_{i:i+k})$$

$$T_{SD} = T_{id}^{*} + \underbrace{(T_{id} - T_{id}^{*})}_{\Delta_{\text{partition}}} + \underbrace{(T_{SD} - T_{id})}_{\Delta_{\text{rejection}}}$$

## 3. Acceptance-метрики (пред-существующие)
| Метрика | Описание | Нотация |
|---|---|---|
| `…num_drafts` | Число draft-раундов (per request per spec-step). | $R$ |
| `…num_draft_tokens` | Всего предложено draft-токенов. | $\sum K$ |
| `…num_accepted_tokens` | Всего принято draft-токенов. | $\sum_r A_r$ |
| `…num_accepted_tokens_per_pos` | Принято на каждой позиции (`Vector`). | — |

Accepted length $\mathbb{E}[A]=1+\frac{\text{accepted}}{\text{drafts}}$ (с bonus); acceptance rate $=\frac{\text{accepted}}{\text{draft\_tokens}}$.

## 4. Поток данных
```
SpecDecodeTimer.time_stage(...)  — CUDA events
  begin_step gpu_model_runner.py:4336 ; set_num_verified_positions :4338
  target_forward :4359 ; verify :3626 ; sample :3613 ; draft_total :4525
  draft pos=i llm_base_proposer.py:544,688 ; drain (лаг 1 шаг) :4653
SpecDecodeTimingStats (timing.py) -> ModelRunnerOutput.spec_decode_timing (outputs.py:274)  [pickle-IPC]
scheduler.py:1501 read ; :1819 observe_timing (1x/шаг)
  -> SpecDecodingProm.observe (metrics.py:398) -> µs Counters + Vector
  -> SpecDecodingLogging._log_timing (metrics.py:175) -> лог-строка
  -> reader.py:109 -> get_metrics()
```

## 5. Оговорки
- **Лаг 1 шаг** (double-buffer): тайминг шага N экспортируется на N+1; на границах prefill/decode теряется ≤1 сэмпл/шаг-без-драфтов.
- **Warmup/JIT**: первые шаги включают JIT Triton-ядер → раздувают средние `verify`/`draft_total`.
- **µs-int**: `Vector.values: list[int]` → тайминг в целых µs.
- **V1-only**: инструментировано в V1 model runner; флаг форсит V1.

## 6. Планируется (ещё НЕ реализовано)
| Величина | Описание | Статус |
|---|---|---|
| $T_T(0)$ baseline, $\eta(K)$ | Из бинов `target_forward_*_by_positions` (bin[1]/bin[K+1]). | ✅ Реализовано |
| $\alpha_k,\beta_k$ cost-model | Фит $T_{T,r}\approx\alpha\,\bar U_r+\beta$ по бинам $k$. | Tier 3 (оффлайн) |
| $T_{SD}=T_{id}^{*}+\Delta_{\text{part}}+\Delta_{\text{rej}}$ | Bellman-DP + ideal-rerun. | Tier 3 (ждёт A1–A4) |

См. также `SPEC_DECODE_TIMING_PLAN.md`, `SPEC_DECODE_OPEN_QUESTIONS.md`.
