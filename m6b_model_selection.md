# M6-B 验证集模型选择

生成时间：2026-10-24T17:19:15+00:00  
基线 commit：`750270014691f1b310759cc8fb68fd07747bf2b8`

**选择：B5 T3 + Regularized FiLM (`b5_t3_film_reg`)**

理由：B5 T3 + Regularized FiLM improves mean by 0.0067 and S3 gap is +0.0015; selected for formal M6-B.

| 变体 | S1 seed42 | S3 fold0 | 均值 | 参数量 |
| --- | ---: | ---: | ---: | ---: |
| B1 T3 Fourier | 0.0798 | 0.1662 | 0.1230 | 498,818 |
| B2 T3 + Mono | 0.0818 | 0.1687 | 0.1252 | 548,227 |
| B3 T3 + Mono + Boundary | 0.0809 | 0.1665 | 0.1237 | 548,227 |
| B4 T3 + Mono + Boundary + Smooth | 0.1124 | 0.1693 | 0.1409 | 548,227 |
| B5 T3 + Regularized FiLM | 0.0649 | 0.1677 | 0.1163 | 535,682 |

选择只使用验证集，未读取测试标签。
