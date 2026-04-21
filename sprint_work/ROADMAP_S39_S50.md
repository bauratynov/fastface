# Roadmap S39 -> S50 — квалити + скорость после S38 ship-quality

**Текущее состояние (S38, tag `s38-ship-quality`, commit `471edd4e`):**
- cos-sim 0.986415 (ship-порог 0.98 пройден)
- speed 13.4-14.3 ms, ~2.3x ORT
- 20/20 wins в стабильном бенче
- 96 KB exe + 42 MB weights
- Backward compat FFW3 сохранён

## Приоритеты S39-S50 (12 sprints)

### Quality moonshots
| Sprint | Цель | Ожидание | Риск |
|---|---|---|---|
| S39 | Калибровка на 100 faces вместо 20 | 0.986 -> 0.988-0.990 | Нет |
| S43 | Asymmetric per-channel (zero points) | 0.986 -> 0.980 (S21 пробовали, чуть хуже) | Моя низкая уверенность |
| S46 | INT16 для первого/последнего Conv (mixed precision) | 0.986 -> 0.99+ | Потенциально низкая скорость |

### Speed moonshots
| Sprint | Цель | Ожидание | Риск |
|---|---|---|---|
| S40 | Стабильный бенч S38 (finalize) | декисивные цифры | Нет |
| S41 | BN+ADD+Conv triple-fusion | 14 ms -> 12-13 ms | Средний |
| S42 | Direct 1x1 conv (skip im2col) | 14 ms -> 13-13.5 ms | Низкий |
| S44 | Batched INT8 foundation | INT8 B=8 inference | Средний |
| S45 | Batched INT8 perf | <10 ms/face at B=8? | Средний |

### Product/commercial
| Sprint | Цель |
|---|---|
| S47 | Consolidation + regression suite |
| S48 | ARM NEON первый порт |
| S49 | Face detector stub (RetinaFace/YOLO-Face INT8) |
| S50 | Full writeup + comprehensive bench |

## S39 план подробно
- Re-run `export_op_scales_v2.py` с `N_CALIB=100` (у нас 13К LFW изображений).
- Re-run `prepare_weights_v3.py` для FFW4.
- Бенч cos-sim. Если >0.99 — ещё большее увеличение при том же перформансе.

## Целевое состояние S50
- cos-sim >= 0.99 (target 0.995)
- speed <= 11 ms на b=1 (3x+ ORT)
- b=8 INT8 <= 7 ms/face
- ARM NEON bench числа для reference
- End-to-end "face → embedding" demo на реальной камере
