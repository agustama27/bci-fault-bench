<!-- T:recon -->
| Condición | d nominal (s) | Cortes | Hueco (s) mediana [mín–máx] | Hueco ia_max (s) mediana | Espera = hueco − d (s) mediana [mín–máx] | Regla lineal: predicción (s) | Lineal: dentro de ±40 ms | Escalonada: predicción (s) | Escalonada: dentro de ±40 ms |
|---|---|---|---|---|---|---|---|---|---|
| disconnect-n2-1 | 1.000 | 90 | 1.52 [1.52–1.56] | 1.560 | 0.52 [0.52–0.56] | 1.52 | 90/90 | 1.52 | 90/90 |
| disconnect-n5-1 | 1.000 | 225 | 1.52 [1.52–1.56] | 1.560 | 0.52 [0.52–0.56] | 1.52 | 225/225 | 1.52 | 225/225 |
| disconnect-n10-1 | 1.000 | 450 | 1.52 [1.52–1.56] | 1.560 | 0.52 [0.52–0.56] | 1.52 | 450/450 | 1.52 | 450/450 |
| disconnect-n20-1 | 1.000 | 900 | 1.52 [1.52–1.56] | 1.560 | 0.52 [0.52–0.56] | 1.52 | 900/900 | 1.52 | 900/900 |
| disconnect-n5-1.25 | 1.250 | 45 | 2.04 [2.04–2.04] | 2.080 | 0.80 [0.76–0.80] | 1.77 | 0/45 | 2.02 | 45/45 |
| disconnect-n5-1.75 | 1.750 | 45 | 2.52 [2.52–2.52] | 2.559 | 0.76 [0.76–0.80] | 2.27 | 0/45 | 2.52 | 45/45 |
| disconnect-n5-2.25 | 2.250 | 45 | 3.04 [3.04–3.04] | 3.079 | 0.80 [0.76–0.80] | 2.77 | 0/45 | 3.02 | 45/45 |

<!-- T:recon_global -->
| Campaña | Modelo | Cortes | MAE (ms) | Error máx. (ms) | Dentro de ±40 ms (%) |
|---|---|---|---|---|---|
| campaña nueva | lineal max(1,52; d+0,52) | 1800 | 19.86 | 270.00 | 92.50 |
| campaña nueva | escalonado max(1,52; 0,52+0,5*ceil(d/0,5)) | 1800 | 1.11 | 40.00 | 100.00 |
| campaña anterior | lineal max(1,52; d+0,52) | 7154 | 0.21 | 40.00 | 100.00 |
| campaña anterior | escalonado max(1,52; 0,52+0,5*ceil(d/0,5)) | 7154 | 0.21 | 40.00 | 100.00 |

<!-- T:recon_ia -->
| Condición | ia_max sin corregir | ia_max − 0,04 s | hueco_ts (gap_s) |
|---|---|---|---|
| disconnect-n10-1 | 48.9 | 99.8 | 100.0 |
| disconnect-n2-1 | 57.8 | 100.0 | 100.0 |
| disconnect-n20-1 | 49.4 | 99.9 | 100.0 |
| disconnect-n5-1 | 46.7 | 99.6 | 100.0 |
| disconnect-n5-1.25 | 0.0 | 0.0 | 0.0 |
| disconnect-n5-1.75 | 0.0 | 0.0 | 0.0 |
| disconnect-n5-2.25 | 0.0 | 0.0 | 0.0 |

<!-- T:exp_expo -->
| Condición | K | Tocados, mediana entre sujetos [Q1–Q3] (%) | Tocados, global (%) (k/n) | Modelo K(1,52+4)/387 (%) | Ensayos inválidos | Muestras perdidas (% de la ventana, global) |
|---|---|---|---|---|---|---|
| ref | 0 | 0.0 [0.0–0.0] | 0.0 (0/1080) | 0.0 | 0 | 0.00 |
| disconnect-n2-1 | 2 | 4.2 [0.0–4.2] | 2.4 (26/1080) | 2.9 | 0 | 0.62 |
| disconnect-n5-1 | 5 | 8.3 [8.3–8.3] | 7.6 (82/1080) | 7.1 | 0 | 2.09 |
| disconnect-n10-1 | 10 | 12.5 [12.5–16.7] | 14.4 (156/1080) | 14.3 | 0 | 4.07 |
| disconnect-n20-1 | 20 | 29.2 [25.0–29.2] | 27.4 (296/1080) | 28.5 | 0 | 7.66 |

<!-- T:exp_ba -->
| Decodificador | Condición | BA mediana | Δ vs ref, mediana | Δ vs ref, media | Sujetos peor/igual/mejor | p Wilcoxon | p Holm | r | Friedman p |
|---|---|---|---|---|---|---|---|---|---|
| CSP+LDA | ref | 0.7083 |  |  |  |  |  |  |  |
| CSP+LDA | disconnect-n2-1 | 0.7083 | +0.0000 | +0.0000 | 0/9/0 | n.d. | 1.000 | n.d. | 0.478 |
| CSP+LDA | disconnect-n5-1 | 0.7083 | +0.0000 | -0.0046 | 1/8/0 | 1.000 | 1.000 | 0.00 | 0.478 |
| CSP+LDA | disconnect-n10-1 | 0.7083 | +0.0000 | -0.0046 | 1/8/0 | 1.000 | 1.000 | 0.00 | 0.478 |
| CSP+LDA | disconnect-n20-1 | 0.7083 | +0.0000 | -0.0139 | 2/7/0 | 0.500 | 1.000 | 0.22 | 0.478 |
| EEGNet | ref | 0.7500 |  |  |  |  |  |  |  |
| EEGNet | disconnect-n2-1 | 0.7500 | +0.0000 | +0.0000 | 0/9/0 | n.d. | 1.000 | n.d. | 0.671 |
| EEGNet | disconnect-n5-1 | 0.7500 | +0.0000 | +0.0046 | 0/7/2 | 0.500 | 1.000 | 0.22 | 0.671 |
| EEGNet | disconnect-n10-1 | 0.7083 | +0.0000 | -0.0185 | 3/4/2 | 0.312 | 1.000 | 0.34 | 0.671 |
| EEGNet | disconnect-n20-1 | 0.7500 | -0.0417 | -0.0139 | 5/0/4 | 1.000 | 1.000 | 0.00 | 0.671 |

<!-- T:exp_cambio -->
| Condición | Decodificador | Grupo | Ensayos | Cambios | Cambio (%) | IC95 Wilson (%) | Acertaba→falla | Fallaba→acierta | Acierto cond. (%) | Acierto ref. (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| disconnect-n2-1 | CSP+LDA | tocado | 26 | 0 | 0.0 | 0.0–12.9 | 0 | 0 | 80.8 | 80.8 |
| disconnect-n2-1 | CSP+LDA | no tocado | 1054 | 0 | 0.0 | 0.0–0.4 | 0 | 0 | 75.8 | 75.8 |
| disconnect-n2-1 | EEGNet | tocado | 26 | 4 | 15.4 | 6.1–33.5 | 1 | 3 | 80.8 | 73.1 |
| disconnect-n2-1 | EEGNet | no tocado | 1054 | 0 | 0.0 | 0.0–0.4 | 0 | 0 | 77.5 | 77.5 |
| disconnect-n5-1 | CSP+LDA | tocado | 82 | 9 | 11.0 | 5.9–19.6 | 4 | 5 | 73.2 | 72.0 |
| disconnect-n5-1 | CSP+LDA | no tocado | 998 | 0 | 0.0 | 0.0–0.4 | 0 | 0 | 76.3 | 76.3 |
| disconnect-n5-1 | EEGNet | tocado | 82 | 15 | 18.3 | 11.4–28.0 | 5 | 10 | 76.8 | 70.7 |
| disconnect-n5-1 | EEGNet | no tocado | 998 | 0 | 0.0 | 0.0–0.4 | 0 | 0 | 78.0 | 78.0 |
| disconnect-n10-1 | CSP+LDA | tocado | 156 | 22 | 14.1 | 9.5–20.4 | 12 | 10 | 71.8 | 73.1 |
| disconnect-n10-1 | CSP+LDA | no tocado | 924 | 0 | 0.0 | 0.0–0.4 | 0 | 0 | 76.4 | 76.4 |
| disconnect-n10-1 | EEGNet | tocado | 156 | 34 | 21.8 | 16.0–28.9 | 20 | 14 | 74.4 | 78.2 |
| disconnect-n10-1 | EEGNet | no tocado | 924 | 0 | 0.0 | 0.0–0.4 | 0 | 0 | 77.3 | 77.3 |
| disconnect-n20-1 | CSP+LDA | tocado | 296 | 30 | 10.1 | 7.2–14.1 | 15 | 15 | 73.6 | 73.6 |
| disconnect-n20-1 | CSP+LDA | no tocado | 784 | 1 | 0.1 | 0.0–0.7 | 1 | 0 | 76.7 | 76.8 |
| disconnect-n20-1 | EEGNet | tocado | 296 | 53 | 17.9 | 14.0–22.7 | 29 | 24 | 74.3 | 76.0 |
| disconnect-n20-1 | EEGNet | no tocado | 784 | 1 | 0.1 | 0.0–0.7 | 0 | 1 | 78.1 | 77.9 |
| TODAS | CSP+LDA | tocado | 560 | 61 | 10.9 | 8.6–13.7 | 31 | 30 | 73.4 | 73.6 |
| TODAS | CSP+LDA | no tocado | 3760 | 1 | 0.0 | 0.0–0.2 | 1 | 0 | 76.2 | 76.3 |
| TODAS | EEGNet | tocado | 560 | 106 | 18.9 | 15.9–22.4 | 55 | 51 | 75.0 | 75.7 |
| TODAS | EEGNet | no tocado | 3760 | 1 | 0.0 | 0.0–0.2 | 0 | 1 | 77.7 | 77.7 |

<!-- T:exp_dosis -->
| Muestras en la ventana | Decodificador | Ensayos | Cambios | Cambio (%) | IC95 (%) | Acertaba→falla | Fallaba→acierta | Acierto cond. (%) | Acierto ref. (%) |
|---|---|---|---|---|---|---|---|---|---|
| 500-749 | CSP+LDA | 365 | 49 | 13.4 | 10.3–17.3 | 29 | 20 | 71.5 | 74.0 |
| 500-749 | EEGNet | 365 | 82 | 22.5 | 18.5–27.0 | 45 | 37 | 74.0 | 76.2 |
| 750-899 | CSP+LDA | 110 | 9 | 8.2 | 4.4–14.8 | 2 | 7 | 73.6 | 69.1 |
| 750-899 | EEGNet | 110 | 16 | 14.5 | 9.2–22.3 | 6 | 10 | 74.5 | 70.9 |
| 900-999 | CSP+LDA | 85 | 3 | 3.5 | 1.2–9.9 | 0 | 3 | 81.2 | 77.6 |
| 900-999 | EEGNet | 85 | 8 | 9.4 | 4.8–17.5 | 4 | 4 | 80.0 | 80.0 |
| >=1000 | CSP+LDA | 3760 | 1 | 0.0 | 0.0–0.2 | 1 | 0 | 76.2 | 76.3 |
| >=1000 | EEGNet | 3760 | 1 | 0.0 | 0.0–0.2 | 0 | 1 | 77.7 | 77.7 |

<!-- T:ret_ba -->
| Familia | Decodificador | Severidad | BA ref (mediana) | BA cond. (mediana) | Δ mediana | Δ media | Sujetos peor/igual/mejor | p Wilcoxon | p Holm | r | Friedman p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| retención (hold) | CSP+LDA | 10 % | 0.7083 | 0.7500 | +0.0000 | -0.0278 | 4/4/1 | 0.188 | 0.188 | 0.44 | 0.063 |
| retención (hold) | CSP+LDA | 25 % | 0.7083 | 0.7083 | -0.0417 | -0.0556 | 6/2/1 | 0.047 | 0.141 | 0.66 | 0.063 |
| retención (hold) | CSP+LDA | 40 % | 0.7083 | 0.7500 | -0.0833 | -0.0417 | 5/1/3 | 0.086 | 0.172 | 0.57 | 0.063 |
| retención (hold) | EEGNet | 10 % | 0.7500 | 0.7500 | +0.0000 | +0.0046 | 2/4/3 | 0.625 | 0.625 | 0.16 | 0.010 |
| retención (hold) | EEGNet | 25 % | 0.7500 | 0.6667 | -0.0417 | -0.0509 | 8/0/1 | 0.008 | 0.023 | 0.89 | 0.010 |
| retención (hold) | EEGNet | 40 % | 0.7500 | 0.7500 | -0.0833 | -0.0648 | 7/0/2 | 0.051 | 0.102 | 0.65 | 0.010 |
| borrado (burst) | CSP+LDA | 10 % | 0.7083 | 0.7083 | +0.0000 | -0.0139 | 4/4/1 | 0.375 | 0.375 | 0.30 | 0.003 |
| borrado (burst) | CSP+LDA | 25 % | 0.7083 | 0.7500 | -0.0417 | -0.0324 | 5/3/1 | 0.125 | 0.250 | 0.51 | 0.003 |
| borrado (burst) | CSP+LDA | 40 % | 0.7083 | 0.6667 | -0.0417 | -0.0648 | 9/0/0 | 0.004 | 0.012 | 0.96 | 0.003 |
| borrado (burst) | EEGNet | 10 % | 0.7500 | 0.7500 | +0.0000 | -0.0139 | 4/3/2 | 0.562 | 0.594 | 0.19 | 0.007 |
| borrado (burst) | EEGNet | 25 % | 0.7500 | 0.7500 | -0.0417 | -0.0370 | 6/1/2 | 0.297 | 0.594 | 0.35 | 0.007 |
| borrado (burst) | EEGNet | 40 % | 0.7500 | 0.7083 | -0.0833 | -0.0787 | 8/0/1 | 0.008 | 0.023 | 0.89 | 0.007 |

<!-- T:ret_directo -->
| Decodificador | Severidad | n | Δ hold (mediana) | Δ burst (mediana) | Δhold − Δburst (mediana) | p Wilcoxon | p Holm | r | Sujetos con menos caída en hold / más |
|---|---|---|---|---|---|---|---|---|---|
| CSP+LDA | 10 % | 9 | 0.0000 | 0.0000 | 0.0000 | 0.344 | 0.961 | 0.32 | 3/4 |
| CSP+LDA | 25 % | 9 | -0.0417 | -0.0417 | -0.0417 | 0.391 | 0.961 | 0.29 | 2/6 |
| CSP+LDA | 40 % | 9 | -0.0833 | -0.0417 | 0.0000 | 0.320 | 0.961 | 0.33 | 5/3 |
| EEGNet | 10 % | 9 | 0.0000 | 0.0000 | 0.0000 | 0.531 | 1.000 | 0.21 | 4/2 |
| EEGNet | 25 % | 9 | -0.0417 | -0.0417 | 0.0000 | 0.719 | 1.000 | 0.12 | 2/4 |
| EEGNet | 40 % | 9 | -0.0833 | -0.0833 | 0.0000 | 0.469 | 1.000 | 0.24 | 4/2 |

<!-- T:ret_cambio -->
| Familia | Decodificador | Severidad | Grupo | Ensayos | Cambios | Cambio (%) | IC95 (%) | Acertaba→falla | Fallaba→acierta | Acierto cond. (%) | Acierto ref. (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hold | CSP+LDA | 10 % | tocado | 1080 | 62 | 5.7 | 4.5–7.3 | 33 | 29 | 75.6 | 75.9 |
| hold | EEGNet | 10 % | tocado | 1080 | 93 | 8.6 | 7.1–10.4 | 46 | 47 | 77.5 | 77.4 |
| hold | CSP+LDA | 25 % | tocado | 1080 | 133 | 12.3 | 10.5–14.4 | 88 | 45 | 71.9 | 75.9 |
| hold | EEGNet | 25 % | tocado | 1080 | 171 | 15.8 | 13.8–18.1 | 105 | 66 | 73.8 | 77.4 |
| hold | CSP+LDA | 40 % | tocado | 1080 | 208 | 19.3 | 17.0–21.7 | 132 | 76 | 70.7 | 75.9 |
| hold | EEGNet | 40 % | tocado | 1080 | 243 | 22.5 | 20.1–25.1 | 158 | 85 | 70.6 | 77.4 |
| burst | CSP+LDA | 10 % | tocado | 1080 | 65 | 6.0 | 4.7–7.6 | 37 | 28 | 75.1 | 75.9 |
| burst | EEGNet | 10 % | tocado | 1080 | 85 | 7.9 | 6.4–9.6 | 51 | 34 | 75.8 | 77.4 |
| burst | CSP+LDA | 25 % | tocado | 1080 | 97 | 9.0 | 7.4–10.8 | 59 | 38 | 74.0 | 75.9 |
| burst | EEGNet | 25 % | tocado | 1080 | 163 | 15.1 | 13.1–17.4 | 103 | 60 | 73.4 | 77.4 |
| burst | CSP+LDA | 40 % | tocado | 1080 | 151 | 14.0 | 12.0–16.2 | 106 | 45 | 70.3 | 75.9 |
| burst | EEGNet | 40 % | tocado | 1080 | 261 | 24.2 | 21.7–26.8 | 171 | 90 | 69.9 | 77.4 |

<!-- T:ret_cambio_sujeto -->
| Decodificador | Severidad | n | Cambio hold (mediana entre sujetos) | Cambio burst (mediana entre sujetos) | Dif. (mediana) | p Wilcoxon | p Holm | r |
|---|---|---|---|---|---|---|---|---|
| CSP+LDA | 10 % | 9 | 0.0500 | 0.0417 | 0.0083 | 0.672 | 1.000 | 0.14 |
| CSP+LDA | 25 % | 9 | 0.1000 | 0.0833 | 0.0250 | 0.188 | 0.562 | 0.44 |
| CSP+LDA | 40 % | 9 | 0.1250 | 0.1250 | 0.0083 | 0.578 | 1.000 | 0.19 |
| EEGNet | 10 % | 9 | 0.0917 | 0.0500 | 0.0083 | 0.570 | 1.000 | 0.19 |
| EEGNet | 25 % | 9 | 0.1167 | 0.1333 | 0.0000 | 0.641 | 1.000 | 0.16 |
| EEGNet | 40 % | 9 | 0.2083 | 0.2583 | -0.0083 | 0.215 | 0.645 | 0.41 |

<!-- T:sil_cond -->
| Campaña | Condición | Evaluables (s) | Degradados (s) | Operativos y degradados (s) | Proporción por ejecución, mediana entre sujetos | Máx. entre sujetos | Global (%) | Degradado que es silencioso (%) | Ejecuciones con ≥1 s (de 45) | Sujetos con ≥1 s (de 9) | Segundos sin muestras |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nueva | ref | 12750 | 0 | 0 | 0.0000 | 0.0000 | 0.00 | n.d. | 0 | 0 [] | 0 |
| nueva | hold_trial-0.1 | 12750 | 40 | 40 | 0.0000 | 0.0000 | 0.31 | 100.0 | 2 | 1 [5] | 0 |
| nueva | hold_trial-0.25 | 12750 | 416 | 416 | 0.0000 | 0.0935 | 3.26 | 100.0 | 8 | 4 [4, 5, 7, 9] | 0 |
| nueva | hold_trial-0.4 | 12750 | 341 | 341 | 0.0000 | 0.0000 | 2.67 | 100.0 | 5 | 5 [1, 5, 7, 8, 9] | 0 |
| anterior | ref | 12750 | 0 | 0 | 0.0000 | 0.0000 | 0.00 | n.d. | 0 | 0 [] | 0 |
| anterior | burst_trial-0.1 | 12750 | 124 | 124 | 0.0000 | 0.0000 | 0.97 | 100.0 | 4 | 3 [5, 6, 7] | 0 |
| anterior | burst_trial-0.25 | 12750 | 370 | 369 | 0.0000 | 0.0839 | 2.89 | 99.7 | 10 | 6 [1, 3, 4, 5, 7, 9] | 13 |
| anterior | burst_trial-0.4 | 12750 | 465 | 436 | 0.0000 | 0.0000 | 3.42 | 93.8 | 8 | 6 [3, 4, 5, 6, 7, 9] | 1308 |

<!-- T:sil_otras -->
| Condición | Evaluables (s) | Degradados (s) | Operativos y degradados (s) | Mediana entre sujetos | Ejecuciones con ≥1 s | Segundos sin muestras |
|---|---|---|---|---|---|---|
| disconnect-n10-1 | 12750 | 27 | 26 | 0.0000 | 2 | 246 |
| disconnect-n2-1 | 12750 | 0 | 0 | 0.0000 | 0 | 56 |
| disconnect-n20-1 | 12750 | 0 | 0 | 0.0000 | 0 | 501 |
| disconnect-n5-1 | 12750 | 0 | 0 | 0.0000 | 0 | 132 |
| disconnect-n5-1.25 | 2822 | 0 | 0 | 0.0000 | 0 | 48 |
| disconnect-n5-1.75 | 2822 | 16 | 16 | 0.0000 | 1 | 72 |
| disconnect-n5-2.25 | 2822 | 0 | 0 | 0.0000 | 0 | 91 |

<!-- T:sil_tele -->
| Variable | Severidad | Ref (mediana entre sujetos) | Hold (mediana entre sujetos) | Diferencia | p Wilcoxon | p Holm | r |
|---|---|---|---|---|---|---|---|
| recv_ratio | 10 % | 0.99984 | 0.99984 | 0.00000 | n.d. | 1.000 | n.d. |
| recv_ratio | 25 % | 0.99984 | 0.99984 | 0.00000 | n.d. | 1.000 | n.d. |
| recv_ratio | 40 % | 0.99984 | 0.99984 | 0.00000 | n.d. | 1.000 | n.d. |
| n_gaps | 10 % | 0.00000 | 0.00000 | 0.00000 | n.d. | 1.000 | n.d. |
| n_gaps | 25 % | 0.00000 | 0.00000 | 0.00000 | n.d. | 1.000 | n.d. |
| n_gaps | 40 % | 0.00000 | 0.00000 | 0.00000 | n.d. | 1.000 | n.d. |
| gap_s | 10 % | 0.00000 | 0.00000 | 0.00000 | n.d. | 1.000 | n.d. |
| gap_s | 25 % | 0.00000 | 0.00000 | 0.00000 | n.d. | 1.000 | n.d. |
| gap_s | 40 % | 0.00000 | 0.00000 | 0.00000 | n.d. | 1.000 | n.d. |
| lat_mean_ms | 10 % | 2.67496 | 2.69977 | 0.02509 | 0.004 | 0.012 | 0.96 |
| lat_mean_ms | 25 % | 2.67496 | 2.69863 | 0.02344 | 0.004 | 0.012 | 0.96 |
| lat_mean_ms | 40 % | 2.67496 | 2.69339 | 0.01584 | 0.004 | 0.012 | 0.96 |
| lat_max_ms | 10 % | 20.12000 | 16.15000 | 4.01000 | 1.000 | 1.000 | 0.00 |
| lat_max_ms | 25 % | 20.12000 | 20.15000 | -1.96000 | 0.820 | 1.000 | 0.08 |
| lat_max_ms | 40 % | 20.12000 | 24.14000 | 4.00000 | 0.195 | 0.586 | 0.43 |
| ia_std_ms | 10 % | 1.88917 | 1.88806 | -0.00336 | 0.910 | 1.000 | 0.04 |
| ia_std_ms | 25 % | 1.88917 | 1.88284 | -0.00165 | 0.496 | 1.000 | 0.23 |
| ia_std_ms | 40 % | 1.88917 | 1.88920 | -0.00426 | 0.820 | 1.000 | 0.08 |
| ia_max_ms | 10 % | 42.84000 | 42.52000 | -0.32000 | 0.020 | 0.059 | 0.78 |
| ia_max_ms | 25 % | 42.84000 | 42.65000 | -0.10000 | 0.652 | 0.652 | 0.15 |
| ia_max_ms | 40 % | 42.84000 | 42.66000 | -0.21000 | 0.164 | 0.328 | 0.46 |
| secs_no_data | 10 % | 0.00000 | 0.00000 | 0.00000 | n.d. | 1.000 | n.d. |
| secs_no_data | 25 % | 0.00000 | 0.00000 | 0.00000 | n.d. | 1.000 | n.d. |
| secs_no_data | 40 % | 0.00000 | 0.00000 | 0.00000 | n.d. | 1.000 | n.d. |

<!-- T:sil_det -->
| Grupo de segundos | Segundos | Alarmas | Tasa de alarma (%) | Degradados | Alarmas en degradados | Recall sobre degradados (%) | Silenciosos | Alarmas en silenciosos |
|---|---|---|---|---|---|---|---|---|
| ref nueva (LOSO) | 12750 | 340 | 2.7 | 0 | 0 | n.d. | 0 | 0 |
| hold_trial-0.1 (todos los segundos) | 12750 | 310 | 2.4 | 40 | 1 | 2.5 | 40 | 1 |
| hold_trial-0.25 (todos los segundos) | 12750 | 352 | 2.8 | 416 | 11 | 2.6 | 416 | 11 |
| hold_trial-0.4 (todos los segundos) | 12750 | 352 | 2.8 | 341 | 9 | 2.6 | 341 | 9 |
| hold (3 severidades, todos los segundos) | 38250 | 1014 | 2.7 | 797 | 21 | 2.6 | 797 | 21 |
| hold (segundos con tramo) | 9364 | 195 | 2.1 | 220 | 9 | 4.1 | 220 | 9 |
| ref anterior (LOSO) | 12750 | 334 | 2.6 | 0 | 0 | n.d. | 0 | 0 |
| burst_trial-0.1 (todos los segundos) | 12750 | 2465 | 19.3 | 124 | 20 | 16.1 | 124 | 20 |
| burst_trial-0.25 (todos los segundos) | 12750 | 3375 | 26.5 | 370 | 98 | 26.5 | 369 | 97 |
| burst_trial-0.4 (todos los segundos) | 12750 | 4271 | 33.5 | 465 | 147 | 31.6 | 436 | 118 |
| burst (3 severidades, todos los segundos) | 38250 | 10111 | 26.4 | 959 | 265 | 27.6 | 929 | 235 |

<!-- T:sil_auc -->
| variable | burst (anterior), todos los segundos | hold, segundos con tramo | hold, todos los segundos |
|---|---|---|---|
| exceptions | n.d. | n.d. | n.d. |
| gap_s | 0.574 | n.d. | n.d. |
| ia_max | 0.543 | 0.527 | 0.512 |
| ia_std | 0.553 | 0.519 | 0.509 |
| lat_max | 0.546 | 0.559 | 0.553 |
| lat_mean | 0.563 | 0.553 | 0.547 |
| n_gaps | 0.562 | n.d. | n.d. |
| recv_ratio | 0.623 | 0.502 | 0.500 |

<!-- T:ctl_dec -->
| Decodificador | Ejecuciones | Ensayos | Decisiones idénticas | % idénticas | Ejecuciones con alguna diferencia | % n_samples igual | |Δ proba| media |
|---|---|---|---|---|---|---|---|
| CSP+LDA | 45 | 1080 | 1080 | 100.0000 | 0 | 99.9074 | 0.0004 |
| EEGNet | 45 | 1080 | 1080 | 100.0000 | 0 | 100.0000 | 0.0000 |

<!-- T:ctl_tele -->
| Variable | Nueva (mediana entre sujetos) | Anterior (mediana entre sujetos) | Diferencia (mediana) | Dif. por ejecución mín. | Dif. por ejecución máx. | p Wilcoxon | r | Sujetos mayor/menor en la nueva |
|---|---|---|---|---|---|---|---|---|
| lat_mean_ms | 2.67496 | 2.66377 | 0.01561 | -0.01517 | 0.06227 | 0.008 | 0.89 | 8/1 |
| lat_max_ms | 20.12000 | 9.36000 | -1.75000 | -30.88000 | 30.92000 | 0.910 | 0.04 | 3/6 |
| ia_std_ms | 1.88917 | 1.88850 | 0.00486 | -0.06930 | 0.07276 | 0.820 | 0.08 | 5/4 |
| ia_max_ms | 42.84000 | 42.57000 | 0.47000 | -3.47000 | 5.20000 | 0.121 | 0.52 | 8/1 |
| recv_ratio | 0.99984 | 0.99984 | 0.00000 | 0.00000 | 0.00000 | n.d. | n.d. | 0/0 |
| decision_delay_ms | 1031.75000 | 1031.75000 | -0.00000 | -2.40000 | 3.05000 | 1.000 | 0.00 | 4/5 |
| bacc | 0.70833 | 0.70833 | 0.00000 | 0.00000 | 0.00000 | n.d. | n.d. | 0/0 |
| valid_rate | 1.00000 | 1.00000 | 0.00000 | 0.00000 | 0.00000 | n.d. | n.d. | 0/0 |
| timing_p99_ms | 0.00155 | 0.00136 | 0.00017 | -0.00010 | 0.00064 | 0.004 | 0.96 | 9/0 |
| bacc EEGNet (ceros vs E2) | 0.75000 | 0.75000 | 0.00000 | 0.00000 | 0.00000 | n.d. | n.d. | 0/0 |

