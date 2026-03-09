# **Evidence suggests Jain’s Fairness Index is limited for fine‑grained spatial equity in grids**

**Figure 1:** Consensus on Jain index as fairness metric

**Overall:** Jain’s Fairness Index (JFI) is a widely accepted, axiomatic scalar measure of distributional fairness, but by itself it does not fully capture multi-dimensional or spatial aspects of equity in discrete grids.

## **What Jain’s Index Captures Well**

* JFI is **scale‑independent, bounded, and symmetric**, and satisfies key fairness axioms such as continuity and population‑independence, making it a robust generic measure of how evenly a resource vector is shared (Lan et al. 2009, 1-9; Jain et al. 1998).  
* It is **Schur‑concave**, increasing when allocations become more equal, which matches intuitive ideas of fairness in many network resource problems (Lan et al. 2009, 1-9).  
* In integer assignment and energy/grid applications, embedding JFI in optimization yields solutions that simultaneously reflect efficiency and a notion of fairness across agents or locations (Sediq et al. 2013, 3496-3509; Rezaeinia et al. 2023, 1-23; AbdulSamadElSkaff et al. 2025, 1-6; Zarabie et al. 2018, 6029-6040).

## **Known Limitations Relevant to Spatial Grids**

* Fairness is **multi-conceptual** (max–min, proportionality, envy‑freeness, etc.); different indices can rate the same allocation differently, so any single index (including JFI) captures only one notion of fairness (Bisio and Marchese 2014, 8-14).  
* JFI is **agnostic to geometry**: it evaluates the allocation vector, not the spatial arrangement of cells, so it cannot distinguish, for example, clustered deprivation vs. checkerboard patterns with identical per‑cell resources (Bisio and Marchese 2014, 8-14).  
* Multi-criteria work shows that to represent properties like proportionality, envy‑freeness, and equity, **multiple JFI-like features or indices are needed**, not a single scalar (Köppen et al. 2013, 841-846).  
* Studies of spatial equity in health, parks, education and transport typically rely on **Gini/Theil, Lorenz curves, accessibility indices, and agglomeration measures**, not JFI, precisely to capture spatial distribution and access patterns (Zhang and Li 2024; Azmoodeh et al. 2021, 2790 \- 2807; Pu 2021; Feitosa et al. 2021, 674-683; Lu et al. 2022; Liu et al. 2021; Cheng et al. 2022; Yu et al. 2021).

### **Summary Table: JFI vs. Spatial Equity Needs**

| Aspect | JFI Handles | Spatial Grid Needs | Citations |
| ----- | ----- | ----- | ----- |
| Overall evenness | Yes (scalar fairness) | Useful but insufficient | (Lan et al. 2009, 1-9; Sediq et al. 2013, 3496-3509; Jain et al. 1998\) |
| Spatial configuration | No | Need distance / adjacency modeling | (Bisio and Marchese 2014, 8-14; Zhang and Li 2024; Liu et al. 2021\) |
| Multi-dimensional needs (access, vulnerability) | Only via extensions | Often multiple indicators | (Köppen et al. 2013, 841-846; Azmoodeh et al. 2021, 2790 \- 2807; Pu 2021; Liu et al. 2021\) |

**Figure 2:** Comparison of Jain index and spatial needs

## **Conclusion**

Jain’s Fairness Index is a sound, axiomatic measure of **how equal** a resource vector is, but on discrete spatial grids it only reflects non-spatial evenness. Accurate representation of **spatial equity** typically requires complementing JFI with spatial and multi-criteria measures (e.g., accessibility, Gini/Theil, clustering statistics).

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

T. Lan, D. Kao, M. Chiang and A. Sabharwal. "An Axiomatic Theory of Fairness in Network Resource Allocation." *2010 Proceedings IEEE INFOCOM* (2009): 1-9. [https://doi.org/10.1109/infcom.2010.5461911](https://doi.org/10.1109/infcom.2010.5461911)

M. Köppen, K. Ohnishi and M. Tsuru. "Multi-Jain Fairness Index of Per-Entity Allocation Features for Fair and Efficient Allocation of Network Resources." *2013 5th International Conference on Intelligent Networking and Collaborative Systems* (2013): 841-846. [https://doi.org/10.1109/incos.2013.161](https://doi.org/10.1109/incos.2013.161)

A. B. Sediq, R. Gohary, R. Schoenen and H. Yanikomeroglu. "Optimal Tradeoff Between Sum-Rate Efficiency and Jain's Fairness Index in Resource Allocation." *IEEE Transactions on Wireless Communications*, 12 (2013): 3496-3509. [https://doi.org/10.1109/twc.2013.061413.121703](https://doi.org/10.1109/twc.2013.061413.121703)

I. Bisio and M. Marchese. "The concept of fairness: Definitions and use in bandwidth allocation applied to satellite environment." *IEEE Aerospace and Electronic Systems Magazine*, 29 (2014): 8-14. [https://doi.org/10.1109/maes.2014.6805361](https://doi.org/10.1109/maes.2014.6805361)

R. Jain, D. Chiu and W. Hawe. "A Quantitative Measure Of Fairness And Discrimination For Resource Allocation In Shared Computer Systems." *ArXiv*, cs.NI/9809099 (1998).

Nahid Rezaeinia, J. Goez and Mario Guajardo. "On efficiency and the Jain’s fairness index in integer assignment problems." *Computational Management Science*, 20 (2023): 1-23. [https://doi.org/10.1007/s10287-023-00477-9](https://doi.org/10.1007/s10287-023-00477-9)

Jun Zhang and Jiawei Li. "Study on the Spatial Arrangement of Urban Parkland under the Perspective of Equity—Taking Harbin Main City as an Example." *Land* (2024). [https://doi.org/10.3390/land13020248](https://doi.org/10.3390/land13020248)

M. Azmoodeh, F. Haghighi and H. Motieyan. "Proposing an integrated accessibility-based measure to evaluate spatial equity among different social classes." *Environment and Planning B: Urban Analytics and City Science*, 48 (2021): 2790 \- 2807\. [https://doi.org/10.1177/2399808321991543](https://doi.org/10.1177/2399808321991543)

Lida Pu. "Fairness of the Distribution of Public Medical and Health Resources." *Frontiers in Public Health*, 9 (2021). [https://doi.org/10.3389/fpubh.2021.768728](https://doi.org/10.3389/fpubh.2021.768728)

F. Feitosa, Jan Wolf and J. Marques. "Spatial Justice Models: An Exploratory Analysis on Fair Distribution of Opportunities." \*\* (2021): 674-683. [https://doi.org/10.1007/978-3-030-86960-1\_51](https://doi.org/10.1007/978-3-030-86960-1_51)

Wei Lu, Yuechen Li, Rongkun Zhao, Bo He and Zihua Qian. "Spatial Pattern and Fairness Measurement of Educational Resources in Primary and Middle Schools: A Case Study of Chengdu–Chongqing Economic Circle." *International Journal of Environmental Research and Public Health*, 19 (2022). [https://doi.org/10.3390/ijerph191710840](https://doi.org/10.3390/ijerph191710840)

Yara AbdulSamadElSkaff, Hugo Joudrier, Y. Gaoua and Quoc Tuan Tran. "Towards a Fair Disaggregation in Hierarchical Cloud – Edge Control of Distributed Local Energy Communities." *2025 IEEE Kiel PowerTech* (2025): 1-6. [https://doi.org/10.1109/powertech59965.2025.11180730](https://doi.org/10.1109/powertech59965.2025.11180730)

Bingxi Liu, Yunqing Tian, Mengjie Guo, D. Tran, Abdulfattah.A.Q. Alwah and Dawei Xu. "Evaluating the disparity between supply and demand of park green space using a multi-dimensional spatial equity evaluation framework." *Cities* (2021). [https://doi.org/10.1016/j.cities.2021.103484](https://doi.org/10.1016/j.cities.2021.103484)

Gang Cheng, Lei Guo and Tao Zhang. "Spatial Equity Assessment of Bus Travel Behavior for Pilgrimage: Evidence from Lhasa, Tibet, China." *Sustainability* (2022). [https://doi.org/10.3390/su141710486](https://doi.org/10.3390/su141710486)

Huimin Yu, Shuangyan Yu, D. He and Yuanan Lu. "Equity analysis of Chinese physician allocation based on Gini coefficient and Theil index." *BMC Health Services Research*, 21 (2021). [https://doi.org/10.1186/s12913-021-06348-w](https://doi.org/10.1186/s12913-021-06348-w)

A. K. Zarabie, Sanjoy Das and M. N. Faqiry. "Fairness-Regularized DLMP-Based Bilevel Transactive Energy Mechanism in Distribution Systems." *IEEE Transactions on Smart Grid*, 10 (2018): 6029-6040. [https://doi.org/10.1109/tsg.2019.2895527](https://doi.org/10.1109/tsg.2019.2895527)

\---

# **Overall, Jain’s Fairness Index is a useful but limited metric for resource distribution in multi‑agent competitive games.**

**Figure 1:** Estimated consensus on Jain index usefulness and limits

## **Strengths in Multi‑Agent / Game Settings**

* **Widely adopted, simple, and generic**: JFI is one of the most popular fairness measures in networking and multi‑agent resource allocation because it is easy to compute, scale‑invariant, and interpretable as a single number in \[0,1\] summarizing equity of shares (Lan et al. 2009, 1-9; Köppen et al. 2013, 841-846; Sediq et al. 2012, 577-583).  
* **Empirically aligns with human fairness judgments**: In a game-based one‑to‑many allocation with crowdsourced human ratings, the “fairness index” (JFI) and normalized entropy were found to be the *most expressive and generic* among generic metrics, correlating well with perceived fairness across many distributions (Grappiolo et al. 2013, 176-200).  
* **Effective in engineering trade‑offs**: Studies on wireless networks and edge computing use JFI to evaluate and optimize algorithms, showing that maximizing or improving JFI meaningfully reflects “fairer” outcomes while balancing efficiency (Leng 2025; Cheraghy et al. 2024, 1-5; Rezaeinia et al. 2023, 1-23; Ghazi et al. 2025, 118515-118535; Sediq et al. 2012, 577-583).  
* **Axiomatic grounding**: An axiomatic theory of fairness shows JFI arises as a special case of a family of fairness measures satisfying desirable properties (symmetry, scale invariance, Schur‑concavity), giving it a solid theoretical basis (Lan et al. 2009, 1-9).

## **Key Limitations and Extensions**

* **Single-metric blindness**: JFI only sees the final allocation vector; it does not encode context, needs, rights, or protected attributes, which are central in many MAS fairness notions (Fossati et al. 2017, 1-9; Köppen et al. 2013, 841-846; Malfa et al. 2024; Malfa et al. 2025; Zhang and Shah 2014, 2636-2644; Castelnovo et al. 2021; Deldjoo et al. 2021, 457 \- 511; Amigó et al. 2023, 103115).  
* **Can be misleading for multi‑dimensional or strategic settings**: Work on multi‑metric “Multi‑Jain” indices shows that JFI on a single observable (e.g., throughput) can miss proportionality, envy‑freeness, or equity; multiple feature-wise JFIs are needed to better capture justness (Köppen et al. 2013, 841-846).  
* **Not tailored to group/algorithmic fairness**: Group‑based notions (demographic parity, counterfactual fairness, conditional statistical parity) capture discrimination patterns that JFI cannot express (Malfa et al. 2024; Malfa et al. 2025; Castelnovo et al. 2021; Deldjoo et al. 2021, 457 \- 511; Amigó et al. 2023, 103115).  
* **Motivating alternative indices**: Game‑theoretic “mood value” approaches adapt JFI to individual satisfaction rates, arguing classical JFI and related notions are limited when agents know others’ demands and total resources (Fossati et al. 2017, 1-9; Fossati et al. 2018, 2801-2814).

## **Practical Takeaway**

Jain’s Fairness Index is **effective as a baseline, scalar equity measure** in multi‑agent competitive games, especially for comparing algorithms or tuning efficiency–fairness trade‑offs. However, it should be combined with **context-aware or group‑aware metrics** (needs, protected attributes, satisfaction, multi‑dimensional fairness) rather than used as the sole notion of fairness.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Xiutong Leng. "Strategic Learning in Multi-Player Bandit Problems: A Game-Theoretic Approach to Resource Allocation in Edge Computing." *ITM Web of Conferences* (2025). [https://doi.org/10.1051/itmconf/20257803027](https://doi.org/10.1051/itmconf/20257803027)

Francesca Fossati, Stefano Moretti and Stefano Secci. "A mood value for fair resource allocations." *2017 IFIP Networking Conference (IFIP Networking) and Workshops* (2017): 1-9. [https://doi.org/10.23919/ifipnetworking.2017.8264839](https://doi.org/10.23919/ifipnetworking.2017.8264839)

T. Lan, D. Kao, M. Chiang and A. Sabharwal. "An Axiomatic Theory of Fairness in Network Resource Allocation." *2010 Proceedings IEEE INFOCOM* (2009): 1-9. [https://doi.org/10.1109/infcom.2010.5461911](https://doi.org/10.1109/infcom.2010.5461911)

Maryam Cheraghy, Meysam Soltanpour, Belal Abuhaija, H. B. Abdalla and Kennedy Ehimwenma. "Game-Theoretic-Based Resource Allocation Algorithm for SCMA Max-Min Problem to Maximize Fairness." *2024 IEEE 7th International Conference on Electronics and Communication Engineering (ICECE)* (2024): 1-5. [https://doi.org/10.1109/icece63871.2024.10976939](https://doi.org/10.1109/icece63871.2024.10976939)

M. Köppen, K. Ohnishi and M. Tsuru. "Multi-Jain Fairness Index of Per-Entity Allocation Features for Fair and Efficient Allocation of Network Resources." *2013 5th International Conference on Intelligent Networking and Collaborative Systems* (2013): 841-846. [https://doi.org/10.1109/incos.2013.161](https://doi.org/10.1109/incos.2013.161)

Corrado Grappiolo, H. P. Martínez and Georgios N. Yannakakis. "Validating Generic Metrics of Fairness in Game-Based Resource Allocation Scenarios with Crowdsourced Annotations." *Trans. Comput. Collect. Intell.*, 13 (2013): 176-200. [https://doi.org/10.1007/978-3-642-54455-2\_8](https://doi.org/10.1007/978-3-642-54455-2_8)

Nahid Rezaeinia, J. Goez and Mario Guajardo. "On efficiency and the Jain’s fairness index in integer assignment problems." *Computational Management Science*, 20 (2023): 1-23. [https://doi.org/10.1007/s10287-023-00477-9](https://doi.org/10.1007/s10287-023-00477-9)

G. Malfa, Jie M. Zhang, Michael Luck and Elizabeth Black. "Using Protected Attributes to Consider Fairness in Multi-Agent Systems." *ArXiv*, abs/2410.12889 (2024). [https://doi.org/10.48550/arxiv.2410.12889](https://doi.org/10.48550/arxiv.2410.12889)

Francesca Fossati, Sahar Hoteit, Stefano Moretti and Stefano Secci. "Fair Resource Allocation in Systems With Complete Information Sharing." *IEEE/ACM Transactions on Networking*, 26 (2018): 2801-2814. [https://doi.org/10.1109/tnet.2018.2878644](https://doi.org/10.1109/tnet.2018.2878644)

G. Malfa, Jie M. Zhang, Michael Luck and Elizabeth Black. "Fairness Aware Reinforcement Learning via Proximal Policy Optimization." *ArXiv*, abs/2502.03953 (2025). [https://doi.org/10.48550/arxiv.2502.03953](https://doi.org/10.48550/arxiv.2502.03953)

Saeedeh Ghazi, Saeed Farzi and A. Nikoofard. "Federated Learning for All: A Reinforcement Learning-Based Approach for Ensuring Fairness in Client Selection." *IEEE Access*, 13 (2025): 118515-118535. [https://doi.org/10.1109/access.2025.3586943](https://doi.org/10.1109/access.2025.3586943)

Chongjie Zhang and J. Shah. "Fairness in Multi-Agent Sequential Decision-Making." \*\* (2014): 2636-2644.

Alessandro Castelnovo, Riccardo Crupi, Greta Greco, D. Regoli, I. Penco and A. Cosentini. "A clarification of the nuances in the fairness metrics landscape." *Scientific Reports*, 12 (2021). [https://doi.org/10.1038/s41598-022-07939-1](https://doi.org/10.1038/s41598-022-07939-1)

Yashar Deldjoo, V. W. Anelli, Hamed Zamani, Alejandro Bellogín and T. D. Noia. "A flexible framework for evaluating user and item fairness in recommender systems." *User Modeling and User-Adapted Interaction*, 31 (2021): 457 \- 511\. [https://doi.org/10.1007/s11257-020-09285-1](https://doi.org/10.1007/s11257-020-09285-1)

Enrique Amigó, Yashar Deldjoo, Stefano Mizzaro and Alejandro Bellogín. "A unifying and general account of fairness measurement in recommender systems." *Inf. Process. Manag.*, 60 (2023): 103115\. [https://doi.org/10.1016/j.ipm.2022.103115](https://doi.org/10.1016/j.ipm.2022.103115)

A. B. Sediq, R. Gohary and H. Yanikomeroglu. "Optimal tradeoff between efficiency and Jain's fairness index in resource allocation." *2012 IEEE 23rd International Symposium on Personal, Indoor and Mobile Radio Communications \- (PIMRC)* (2012): 577-583. [https://doi.org/10.1109/pimrc.2012.6362851](https://doi.org/10.1109/pimrc.2012.6362851)

\---  
**Evidence is insufficient to link Moran’s I in game maps to perceived fairness**

**Figure 1:** Consensus on Moran’s I–fairness link in games

## **What the literature covers (and misses)**

**1\. Moran’s I and spatial patterns (but not player perception)**  
 Moran’s I is well-studied as a **global measure of spatial autocorrelation** and has been refined, extended to 2D functions, and standardized for comparison across datasets (Chen 2022; Chen 2020; Yamada 2024; Shortridge 2007, 362-371; DeWitt et al. 2021, 2897 \- 2918; Zhang et al. 2023, 3355 \- 3368). These works show how I captures clustering/dispersion and its statistical behavior, but none connect it to **human fairness judgments** or game maps.

Work on **local Moran’s I** and visualization focuses on making spatial dependence interpretable for analysts, not for players or fairness perception (Mason et al. 2024, 71-75).

**2\. Spatial fairness in algorithms, not in level design**  
 “Spatial fairness” has emerged around **location-based algorithmic decisions** (e.g., resource allocation, ML predictions), with formal definitions and auditing/learning frameworks to ensure outcomes are not biased by location (He et al. 2024, 4750-4765; Shaham et al. 2022, 167 \- 179; He et al. 2022; Sacharidis et al. 2023, 485-491). These papers measure fairness via statistical independence or equity of outcomes over space, not via **subjective perceived fairness**, and they do not use Moran’s I of a map as a predictor of fairness judgments.

**3\. Fairness in spatial games, but at payoff/strategy level**  
 Evolutionary game models like the **spatial ultimatum game** show that spatial structure and local interactions can promote the evolution of fair strategies and complex patterns (Szolnoki et al. 2012, 078701 ; Szolnoki et al. 2012). However, these works analyze **strategy dynamics and payoff fairness**, not player-perceived fairness of a spatial layout, and do not report Moran’s I of the underlying grids.

### **Summary Table**

| Aspect | What’s studied | Link to your question | Citations |
| ----- | ----- | ----- | ----- |
| Moran’s I | Measurement, modeling, visualization | No perception or game fairness | (Chen 2022; Chen 2020; Yamada 2024; Mason et al. 2024, 71-75; Shortridge 2007, 362-371; DeWitt et al. 2021, 2897 \- 2918; Zhang et al. 2023, 3355 \- 3368\) |
| Spatial fairness | Equity of algorithmic outcomes | Not about map layout or games | (He et al. 2024, 4750-4765; Shaham et al. 2022, 167 \- 179; He et al. 2022; Sacharidis et al. 2023, 485-491) |
| Fairness in spatial games | Strategy/payoff fairness on grids | No use of Moran’s I or map-layout fairness | (Zhang 2024, 129183; Szolnoki et al. 2012, |

     078701

; Szolnoki et al. 2012)|

**Figure 2:** Scope of research vs. map-fairness question

## **Conclusion**

No located studies directly test whether **Moran’s I of a game/map design correlates with players’ perceived fairness**. Existing work gives tools to compute and interpret Moran’s I and to formalize spatial fairness in algorithms, but an empirical link to player fairness perception in level design appears to be an open research question.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Yanguang Chen. "Spatial autocorrelation equation based on Moran’s index." *Scientific Reports*, 13 (2022). [https://doi.org/10.1038/s41598-023-45947-x](https://doi.org/10.1038/s41598-023-45947-x)

Erhu He, Yiqun Xie, Weiye Chen, Sergii Skakun, Han Bao, Rahul Ghosh, Praveen Ravirathinam and Xiaowei Jia. "Learning With Location-Based Fairness: A Statistically-Robust Framework and Acceleration." *IEEE Transactions on Knowledge and Data Engineering*, 36 (2024): 4750-4765. [https://doi.org/10.1109/tkde.2024.3371460](https://doi.org/10.1109/tkde.2024.3371460)

Hong Zhang. "Evolution of cooperation among fairness-seeking agents in spatial public goods game." *Appl. Math. Comput.*, 489 (2024): 129183\. [https://doi.org/10.1016/j.amc.2024.129183](https://doi.org/10.1016/j.amc.2024.129183)

Yanguang Chen. "An analytical process of spatial autocorrelation functions based on Moran’s index." *PLoS ONE*, 16 (2020). [https://doi.org/10.1371/journal.pone.0249589](https://doi.org/10.1371/journal.pone.0249589)

A. Szolnoki, M. Perc and G. Szabó. "Defense mechanisms of empathetic players in the spatial ultimatum game.." *Physical review letters*, 109 7 (2012): 078701\. [https://doi.org/10.1103/physrevlett.109.078701](https://doi.org/10.1103/physrevlett.109.078701)

Hiroshi Yamada. "A New Perspective on Moran’s Coefficient: Revisited." *Mathematics* (2024). [https://doi.org/10.3390/math12020253](https://doi.org/10.3390/math12020253)

Lee Mason, Bl'anaid Hicks and Jonas S. Almeida. "Demystifying Spatial Dependence: Interactive Visualizations for Interpreting Local Spatial Autocorrelation." *2024 IEEE Visualization and Visual Analytics (VIS)* (2024): 71-75. [https://doi.org/10.1109/vis55277.2024.00022](https://doi.org/10.1109/vis55277.2024.00022)

Sina Shaham, Gabriel Ghinita and C. Shahabi. "Models and Mechanisms for Spatial Data Fairness." *Proceedings of the VLDB Endowment. International Conference on Very Large Data Bases*, 16 (2022): 167 \- 179\. [https://doi.org/10.14778/3565816.3565820](https://doi.org/10.14778/3565816.3565820)

Ashton M. Shortridge. "Practical limits of Moran's autocorrelation index for raster class maps." *Comput. Environ. Urban Syst.*, 31 (2007): 362-371. [https://doi.org/10.1016/j.compenvurbsys.2006.07.001](https://doi.org/10.1016/j.compenvurbsys.2006.07.001)

Erhu He, Weiye Chen, Yiqun Xie, Han Bao, Xun Zhou, X. Jia, Zhenling Jiang, Rahul Ghosh and Praveen Ravirathinam. "Sailing in the location-based fairness-bias sphere." *Proceedings of the 30th International Conference on Advances in Geographic Information Systems* (2022). [https://doi.org/10.1145/3557915.3560976](https://doi.org/10.1145/3557915.3560976)

T. DeWitt, J. I. Fuentes, T. Ioerger and M. Bishop. "Rectifying I: three point and continuous fit of the spatial autocorrelation metric, Moran’s I, to ideal form." *Landscape Ecology*, 36 (2021): 2897 \- 2918\. [https://doi.org/10.1007/s10980-021-01256-0](https://doi.org/10.1007/s10980-021-01256-0)

Dimitris Sacharidis, G. Giannopoulos, George Papastefanatos and Kostas Stefanidis. "Auditing for Spatial Fairness." \*\* (2023): 485-491. [https://doi.org/10.48550/arxiv.2302.12333](https://doi.org/10.48550/arxiv.2302.12333)

Ce Zhang, Wangyong Lv, Ping Zhang and Jiacheng Song. "Multidimensional spatial autocorrelation analysis and it’s application based on improved Moran’s I." *Earth Science Informatics*, 16 (2023): 3355 \- 3368\. [https://doi.org/10.1007/s12145-023-01090-9](https://doi.org/10.1007/s12145-023-01090-9)

A. Szolnoki, M. Perc and G. Szabó. "Accuracy in strategy imitations promotes the evolution of fairness in the spatial ultimatum game." *Europhysics Letters*, 100 (2012). [https://doi.org/10.1209/0295-5075/100/28005](https://doi.org/10.1209/0295-5075/100/28005)

\---

**No, greedy hill-climbing in PCG rarely guarantees optimal equilibria, even on symmetric maps.**

**Figure 1:** Consensus that local search rarely guarantees global optima.

## **Why greedy hill-climbing is not enough**

Greedy hill-climbing is a **local search** method: it iteratively improves a map by accepting only better neighbors. On complex PCG landscapes this usually leads to **local optima**, not provably optimal equilibria. General analyses of hill-climbing and generalized hill-climbing show convergence to global optima only under strong conditions (e.g., known global optimum value, particular move rules) that do not hold for typical PCG fitness functions (Johnson and Jacobson 2002, 359-373).

Work on local search for SAT also finds that “neither greediness nor randomness is important” to escape hard regions; performance is dominated by landscape structure and restart/parameter choices, with no guarantee of reaching global optima (Gent and Walsh 1993, 28-33). Similar issues arise in dynamic and engineering optimization, where pure local search tends to stagnate at sub‑optimal solutions unless combined with diversity mechanisms, restarts, or population-based methods (Wang et al. 2009, 763-780; Kaur and Dhillon 2021, 107690; Rodríguez-Esparza et al. 2024, 111784).

## **Evidence from procedural map generation**

In Terra Mystica map generation, **steepest-ascent hill-climbing with random restarts** is competitive but is only reported to find maps satisfying a subset of design constraints, not proven optimal or uniquely balanced equilibria (De Araújo et al. 2020, 163 \- 175). Search-based PCG surveys emphasize **evolutionary/metaheuristic** approaches exactly because landscapes are multi‑modal and designer objectives (e.g., fairness, symmetry, difficulty) conflict, making single-run local search insufficient for global guarantees (Togelius et al. 2011, 172-186; Togelius et al. 2010, 141-150; Shaker et al. 2016, 1-237; Volz et al. 2023, 110121).

Multi-objective map generation explicitly treats balance as part of a **Pareto front**: many trade-off solutions exist, and algorithms explore this front rather than a single “optimal” equilibrium point (Togelius et al. 2010, 3:1-3:8). Symmetry is often enforced by representation or constraints, not by hill-climbing convergence (De Araújo et al. 2020, 163 \- 175; Zafar et al. 2020; Togelius et al. 2010, 3:1-3:8).

## **Conclusion**

Greedy hill-climbing in PCG can produce reasonably balanced symmetric maps, especially with restarts, but there is **no theoretical or empirical basis** to expect it to converge reliably to globally optimal equilibria. For equilibrium-quality guarantees, multi-objective and population-based methods with explicit balance metrics are preferred.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Vanessa Volz, B. Naujoks, P. Kerschke and Tea Tušar. "Tools for Landscape Analysis of Optimisation Problems in Procedural Content Generation for Games." *Appl. Soft Comput.*, 136 (2023): 110121\. [https://doi.org/10.1016/j.asoc.2023.110121](https://doi.org/10.1016/j.asoc.2023.110121)

Luiz Jonatã Pires de Araújo, Alexandr Grichshenko, R. Pinheiro, R. D. Saraiva and Susanna Gimaeva. "Map Generation and Balance in the Terra Mystica Board Game Using Particle Swarm and Local Search." *Advances in Swarm Intelligence*, 12145 (2020): 163 \- 175\. [https://doi.org/10.1007/978-3-030-53956-6\_15](https://doi.org/10.1007/978-3-030-53956-6_15)

J. Togelius, M. Preuss and Georgios N. Yannakakis. "Towards multiobjective procedural map generation." \*\* (2010): 3:1-3:8. [https://doi.org/10.1145/1814256.1814259](https://doi.org/10.1145/1814256.1814259)

J. Togelius, Georgios N. Yannakakis, Kenneth O. Stanley and C. Browne. "Search-Based Procedural Content Generation: A Taxonomy and Survey." *IEEE Transactions on Computational Intelligence and AI in Games*, 3 (2011): 172-186. [https://doi.org/10.1109/tciaig.2011.2148116](https://doi.org/10.1109/tciaig.2011.2148116)

Hongfeng Wang, Dingwei Wang and Shengxiang Yang. "A memetic algorithm with adaptive hill climbing strategy for dynamic optimization problems." *Soft Computing*, 13 (2009): 763-780. [https://doi.org/10.1007/s00500-008-0347-3](https://doi.org/10.1007/s00500-008-0347-3)

J. Togelius, Georgios N. Yannakakis, Kenneth O. Stanley and C. Browne. "Search-Based Procedural Content Generation." \*\* (2010): 141-150. [https://doi.org/10.1007/978-3-642-12239-2\_15](https://doi.org/10.1007/978-3-642-12239-2_15)

Gurpreet Kaur and J. S. Dhillon. "Economic power generation scheduling exploiting hill-climbed Sine-Cosine​ algorithm." *Appl. Soft Comput.*, 111 (2021): 107690\. [https://doi.org/10.1016/j.asoc.2021.107690](https://doi.org/10.1016/j.asoc.2021.107690)

E. Rodríguez-Esparza, Bernardo Morales-Castañeda, Ángel Casas-Ordaz, Diego Oliva, M. Navarro, Arturo Valdivia and Essam H. Houssein. "Handling the balance of operators in evolutionary algorithms through a weighted Hill Climbing approach." *Knowl. Based Syst.*, 294 (2024): 111784\. [https://doi.org/10.1016/j.knosys.2024.111784](https://doi.org/10.1016/j.knosys.2024.111784)

Adeel Zafar, Hasan Mujtaba and M. O. Beg. "Search-based procedural content generation for GVG-LG." *Appl. Soft Comput.*, 86 (2020). [https://doi.org/10.1016/j.asoc.2019.105909](https://doi.org/10.1016/j.asoc.2019.105909)

Ian P. Gent and T. Walsh. "Towards an Understanding of Hill-Climbing Procedures for SAT." \*\* (1993): 28-33.

Noor Shaker, Julian Togelius and Mark J. Nelson. "Procedural Content Generation in Games." \*\* (2016): 1-237. [https://doi.org/10.1007/978-3-319-42716-4](https://doi.org/10.1007/978-3-319-42716-4)

Alan W. Johnson and S. Jacobson. "A class of convergent generalized hill climbing algorithms." *Appl. Math. Comput.*, 125 (2002): 359-373. [https://doi.org/10.1016/s0096-3003(00)00137-5](https://doi.org/10.1016/s0096-3003\(00\)00137-5)

\---

# **Yes, Tabu Search is effective for escaping local optima in procedural map generation.**

**Figure 1:** Overall evidence on Tabu Search effectiveness.

## **Evidence in Procedural Map Generation**

* A Tabu Search (TS)–based generator for **Terra Mystica** maps shows that TS can improve simple hill-climbing by escaping local optima and finding higher-quality layouts; varying tabu list size and neighborhood size allowed the search to move beyond locally good but globally inferior maps (Grichshenko et al. 2020).  
* The study reports TS as a **feasible and effective** method for procedural map generation, producing maps that improve on existing hand-designed ones according to game-specific quality metrics (Grichshenko et al. 2020).

## **How Tabu Search Escapes Local Optima**

* TS is explicitly designed to **relax the local search rule** that “only improving moves are allowed,” by sometimes accepting worsening moves and using a tabu list to prevent returning to recently visited solutions; this directly addresses getting trapped at local optima (Subash et al. 2022; Glover 1990, 74-94; Glover 1989, 4-32).  
* Core TS theory emphasizes using **adaptive memory** to steer the search away from the basin of attraction of recent local optima and toward new regions of the search space (Bentsen et al. 2022, 100028; Hanafi et al. 2023, 1037-1055; Glover 1990, 74-94; Glover 2020).  
* Variants like double-neighborhood TS or exponential-extrapolation memory further speed escape from attraction zones in landscapes with many local optima (Bentsen et al. 2022, 100028; Hanafi et al. 2023, 1037-1055; Amaral et al. 2022, 219-230).

## **Comparison and Hybrids**

* In other domains, combining TS with genetic algorithms improves convergence and solution quality compared with GA alone, by exploiting TS’s strong local search and escape-from-local-optima behavior (Sivaram et al. 2019).  
* Tutorials and reviews of metaheuristics consistently cite TS as a **strong local-search-based metaheuristic** for hard combinatorial layout/route problems, where local optima are pervasive (Osaba et al. 2021; Turgut et al. 2023, 14275-14378; Glover 1990, 74-94; Glover 1989, 4-32).

## **Conclusion**

Overall, research indicates that Tabu Search is well-suited to escaping local optima and has been successfully applied to procedural map generation for games. Its adaptive memory and acceptance of non-improving moves make it a strong choice, especially when carefully tuning tabu list size and neighborhood structure to the specific map encoding.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Alexandr Grichshenko, Luiz Jonatã Pires de Araújo, Susanna Gimaeva and J. A. Brown. "Using Tabu Search Algorithm for Map Generation in the Terra Mystica Tabletop Game." *Proceedings of the 2020 4th International Conference on Intelligent Systems, Metaheuristics & Swarm Intelligence* (2020). [https://doi.org/10.1145/3396474.3396492](https://doi.org/10.1145/3396474.3396492)

Dr. N. subash, M. Ramachandran, Vimala Saravanan and Vidhya Prasanth. "An Investigation on Tabu Search Algorithms Optimization." *Electrical and Automation Engineering* (2022). [https://doi.org/10.46632/1/1/3](https://doi.org/10.46632/1/1/3)

Håkon Bentsen, Arild Hoff and L. M. Hvattum. "Exponential extrapolation memory for tabu search." *EURO J. Comput. Optim.*, 10 (2022): 100028\. [https://doi.org/10.1016/j.ejco.2022.100028](https://doi.org/10.1016/j.ejco.2022.100028)

S. Hanafi, Yang Wang, F. Glover, Wei-Xing Yang and Rick Hennig. "Tabu search exploiting local optimality in binary optimization." *Eur. J. Oper. Res.*, 308 (2023): 1037-1055. [https://doi.org/10.1016/j.ejor.2023.01.001](https://doi.org/10.1016/j.ejor.2023.01.001)

E. Osaba, Esther Villar-Rodriguez, J. Ser, Antonio J. Nebro, D. Molina, A. Latorre, P. Suganthan, C. Coello and Francisco Herrera. "A Tutorial On the design, experimentation and application of metaheuristic algorithms to real-World optimization problems." *ArXiv*, abs/2410.03205 (2021). [https://doi.org/10.1016/j.swevo.2021.100888](https://doi.org/10.1016/j.swevo.2021.100888)

M. Sivaram, K. Batri, A. Mohammed and V. Porkodi. "Exploiting the Local Optima in Genetic Algorithm using Tabu Search." *Indian Journal of Science and Technology* (2019). [https://doi.org/10.17485/ijst/2019/v12i1/139577](https://doi.org/10.17485/ijst/2019/v12i1/139577)

O. Turgut, M. Turgut and Erhan Kırtepe. "A systematic review of the emerging metaheuristic algorithms on solving complex optimization problems." *Neural Computing and Applications*, 35 (2023): 14275-14378. [https://doi.org/10.1007/s00521-023-08481-5](https://doi.org/10.1007/s00521-023-08481-5)

P. Amaral, Ana Mendes and J. M. Espinosa. "A Tabu Search with a Double Neighborhood Strategy." \*\* (2022): 219-230. [https://doi.org/10.1007/978-3-031-10562-3\_16](https://doi.org/10.1007/978-3-031-10562-3_16)

F. Glover. "Tabu Search: A Tutorial." *Interfaces*, 20 (1990): 74-94. [https://doi.org/10.1287/inte.20.4.74](https://doi.org/10.1287/inte.20.4.74)

F. Glover. "Exploiting Local Optimality in Metaheuristic Search." *ArXiv*, abs/2010.05394 (2020).

F. Glover. "Tabu Search \- Part II." *INFORMS J. Comput.*, 2 (1989): 4-32. [https://doi.org/10.1287/ijoc.2.1.4](https://doi.org/10.1287/ijoc.2.1.4)

\---

# **While simulated annealing can avoid poor local optima, it does not always converge better than greedy hill‑climbing; performance depends on the landscape and tuning.**

**Figure 1:** Overall evidence on SA vs hill-climbing convergence.

## **When Simulated Annealing (SA) Helps**

* **Escaping local optima:** Hill‑climbing (HC) is purely exploitative and easily traps in local optima. SA accepts occasional worse moves, improving global exploration in multi‑modal landscapes (Chen et al. 2012, 1-7; Henderson et al. 2006, 287-319; Guilmeau et al. 2021, 101-105; Hajek 1988, 311-329).  
* **Game-like, rugged landscapes:** For discrete, multi‑modal problems (e.g., quadratic assignment, flow shop, TSP—analogous to game balance search), well‑designed SA variants frequently reach better-quality solutions than straightforward hill‑climbers and coordinate‑exchange–style methods, which “quickly converge to local optima” (Adil and Lakhbab 2023; Mao et al. 2024; Franzin and Stützle 2019, 191-206).  
* **Multiobjective or design‑style problems:** In multiobjective and design settings, SA’s ability to explore widely typically yields more diverse and higher‑quality solutions than greedy local methods (Amine 2019, 8134674:1-8134674:13; Mao et al. 2024).

### **Landscape‑Dependent Results**

Generalized hill climbing theory shows that SA and hill‑climbing are both special cases in one framework; which is better depends on **basins of attraction** and landscape structure (Jacobson and Yücesan 2004, 387-405; Johnson and Jacobson 2002, 37-57; Sullivan and Jacobson 2001, 1288-1293). For some landscapes, random‑restart HC (or simple restarts) can match or beat SA in probability of hitting a global optimum, given the same budget (Jacobson and Yücesan 2004, 387-405).

## **Practical Trade‑offs**

| Aspect | Simulated annealing | Greedy hill-climbing | Citations |
| ----- | ----- | ----- | ----- |
| Global exploration | Strong (downhill & uphill moves) | Weak (downhill only) | (Chen et al. 2012, 1-7; Henderson et al. 2006, 287-319; Guilmeau et al. 2021, 101-105; Hajek 1988, 311-329) |
| Sensitivity to local optima | Lower | High | (Chen et al. 2012, 1-7; Henderson et al. 2006, 287-319; Mao et al. 2024; Franzin and Stützle 2019, 191-206) |
| Convergence speed | Often slower, needs tuning | Fast but local | (Adil and Lakhbab 2023; Guilmeau et al. 2021, 101-105; Hajek 1988, 311-329; Orosz and Jacobson 2002, 165-182) |

**Figure 2:** Key convergence trade-offs between SA and hill-climbing.

## **Conclusion**

For game balance optimization on rugged design spaces, **well‑tuned SA is usually more robust against bad local optima than greedy hill‑climbing, but may converge slower and is not uniformly superior**. Performance depends heavily on the payoff landscape, restart strategy, and SA parameter tuning.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Stephen Y. Chen, Carlos Xudiera and James Montgomery. "Simulated annealing with thresheld convergence." *2012 IEEE Congress on Evolutionary Computation* (2012): 1-7. [https://doi.org/10.1109/cec.2012.6256591](https://doi.org/10.1109/cec.2012.6256591)

S. Jacobson and E. Yücesan. "Analyzing the Performance of Generalized Hill Climbing Algorithms." *Journal of Heuristics*, 10 (2004): 387-405. [https://doi.org/10.1023/b:heur.0000034712.48917.a9](https://doi.org/10.1023/b:heur.0000034712.48917.a9)

Alan W. Johnson and S. Jacobson. "On the convergence of generalized hill climbing algorithms." *Discret. Appl. Math.*, 119 (2002): 37-57. [https://doi.org/10.1016/s0166-218x(01)00264-5](https://doi.org/10.1016/s0166-218x\(01\)00264-5)

Khalil Amine. "Multiobjective Simulated Annealing: Principles and Algorithm Variants." *Adv. Oper. Res.*, 2019 (2019): 8134674:1-8134674:13. [https://doi.org/10.1155/2019/8134674](https://doi.org/10.1155/2019/8134674)

N. Adil and H. Lakhbab. "A new improved simulated annealing for traveling salesman problem." *Mathematical Modeling and Computing* (2023). [https://doi.org/10.23939/mmc2023.03.764](https://doi.org/10.23939/mmc2023.03.764)

Darrall Henderson, S. Jacobson and Alan W. Johnson. "The Theory and Practice of Simulated Annealing." \*\* (2006): 287-319. [https://doi.org/10.1007/0-306-48056-5\_10](https://doi.org/10.1007/0-306-48056-5_10)

Yicheng Mao, R. Kessels and Tom C. van der Zanden. "Constructing Bayesian optimal designs for discrete choice experiments by simulated annealing." *Journal of Choice Modelling* (2024). [https://doi.org/10.1016/j.jocm.2025.100551](https://doi.org/10.1016/j.jocm.2025.100551)

Kelly A. Sullivan and S. Jacobson. "A convergence analysis of generalized hill climbing algorithms." *IEEE Trans. Autom. Control.*, 46 (2001): 1288-1293. [https://doi.org/10.1109/9.940936](https://doi.org/10.1109/9.940936)

Thomas Guilmeau, É. Chouzenoux and V. Elvira. "Simulated Annealing: a Review and a New Scheme." *2021 IEEE Statistical Signal Processing Workshop (SSP)* (2021): 101-105. [https://doi.org/10.1109/ssp49050.2021.9513782](https://doi.org/10.1109/ssp49050.2021.9513782)

B. Hajek. "Cooling Schedules for Optimal Annealing." *Math. Oper. Res.*, 13 (1988): 311-329. [https://doi.org/10.1287/moor.13.2.311](https://doi.org/10.1287/moor.13.2.311)

A. Franzin and T. Stützle. "Revisiting simulated annealing: A component-based analysis." *Comput. Oper. Res.*, 104 (2019): 191-206. [https://doi.org/10.1016/j.cor.2018.12.015](https://doi.org/10.1016/j.cor.2018.12.015)

J. E. Orosz and S. Jacobson. "Analysis of Static Simulated Annealing Algorithms." *Journal of Optimization Theory and Applications*, 115 (2002): 165-182. [https://doi.org/10.1023/a:1019633214895](https://doi.org/10.1023/a:1019633214895)

\---

# **Yes, Bayesian optimization can be suitable for high‑dimensional combinatorial board‑game map design, if you use BO methods tailored to discrete, high‑D spaces.**

**Figure 1:** Overall support for BO in high-dimensional combinatorial spaces

## **Evidence from High‑Dimensional Combinatorial BO**

Several recent BO variants explicitly target high‑dimensional combinatorial or mixed discrete spaces:

* **COMBO** models combinatorial spaces as a graph Cartesian product and uses a GP with a diffusion kernel and variable‑selection prior. It is designed for high‑dimensional ordinal/categorical variables and outperforms prior methods on combinatorial benchmarks such as MAXSAT and neural architecture search (Oh et al. 2019, 2910-2920).  
* **Dictionary‑based embeddings** map high‑dimensional binary/categorical structures into low‑dimensional continuous embeddings so standard GP BO can be used; this yields state‑of‑the‑art performance on diverse high‑D combinatorial benchmarks (Deshwal et al. 2023, 7021-7039).  
* **Bounce** introduces nested embeddings for mixed and combinatorial spaces and is explicitly aimed at reliable high‑dimensional BO; it consistently matches or improves on prior state of the art (Papenmeier et al. 2023).  
* **MOCA‑HESP** and related meta‑algorithms further wrap combinatorial BO optimizers (including Bounce) with partitioning strategies to handle very high dimensions (Ngo et al. 2025).  
* General analyses of high‑dimensional BO show that, with appropriate kernels, embeddings, or local‑search strategies, BO can remain effective even in large parameter spaces (Hoang et al. 2025; Jin and Zhan 2025, 80-85; Papenmeier et al. 2025; Wan et al. 2021, 10663-10674; Daulton et al. 2021).

These problems (chip design, NAS, materials, etc.) share key traits with board‑game map design: expensive black‑box evaluation and high‑dimensional discrete design spaces.

## **Relation to Map / Board‑Game PCG**

Existing board‑game and strategy‑map generation work mostly uses **evolutionary and swarm metaheuristics** (GA, PSO, ABC, hill climbing, multi‑objective EA) rather than BO (Alyaseri et al. 2025, 1-33; De Araújo et al. 2020, 163 \- 175; Amiri-Chimeh et al. 2024, 100644; Togelius et al. 2010, 3:1-3:8; Alyaseri 2023; Shaker et al. 2016, 1-237; Silva et al. 2025, 34179 \- 34205). This indicates that BO is **under‑explored**, not unsuitable.

| Question | BO Answer for Map Design | Citations |
| ----- | ----- | ----- |
| High‑D combinatorial variables? | Use COMBO, Bounce, dictionary embeddings, MOCA‑HESP | (Oh et al. 2019, 2910-2920; Papenmeier et al. 2023; Ngo et al. 2025; Deshwal et al. 2023, 7021-7039; Wan et al. 2021, 10663-10674) |
| Expensive evaluations (simulations/playtests)? | BO is sample‑efficient vs. evolutionary baselines | (Oh et al. 2019, 2910-2920; Papenmeier et al. 2023; Deshwal et al. 2023, 7021-7039; Bian et al. 2023, 110630; Daulton et al. 2021\) |
| Multiple design goals (balance, fun, variety)? | Multi‑objective BO (e.g., MORBO) scales to high‑D | (Bian et al. 2023, 110630; Daulton et al. 2021\) |

**Figure 2:** How BO capabilities map to board-game map design needs

## **Conclusion**

Bayesian optimization, in modern high‑dimensional combinatorial variants (COMBO, Bounce, dictionary‑based embeddings, MOCA‑HESP, MORBO), is a **promising and suitable approach** for board‑game map design, especially when evaluations are expensive. In games, evolution and swarm methods are currently more common, so applying these BO methods would be a novel but well‑supported choice.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Sana Alyaseri, Andy M. Connor and Roopak Sinha. "Exploring Metaheuristic Algorithms for Enhanced Game Map Generation in Procedural Content Generation." *Int. J. Appl. Metaheuristic Comput.*, 16 (2025): 1-33. [https://doi.org/10.4018/ijamc.388932](https://doi.org/10.4018/ijamc.388932)

Changyong Oh, J. Tomczak, E. Gavves and M. Welling. "Combinatorial Bayesian Optimization using the Graph Cartesian Product." \*\* (2019): 2910-2920.

Leonard Papenmeier, Luigi Nardi and Matthias Poloczek. "Bounce: Reliable High-Dimensional Bayesian Optimization for Combinatorial and Mixed Spaces." \*\* (2023).

Luiz Jonatã Pires de Araújo, Alexandr Grichshenko, R. Pinheiro, R. D. Saraiva and Susanna Gimaeva. "Map Generation and Balance in the Terra Mystica Board Game Using Particle Swarm and Local Search." *Advances in Swarm Intelligence*, 12145 (2020): 163 \- 175\. [https://doi.org/10.1007/978-3-030-53956-6\_15](https://doi.org/10.1007/978-3-030-53956-6_15)

Lam Ngo, Huong Ha, Jeffrey Chan and Hongyu Zhang. "MOCA-HESP: Meta High-dimensional Bayesian Optimization for Combinatorial and Mixed Spaces via Hyper-ellipsoid Partitioning." *ArXiv*, abs/2508.06847 (2025). [https://doi.org/10.48550/arxiv.2508.06847](https://doi.org/10.48550/arxiv.2508.06847)

Aryan Deshwal, Sebastian Ament, M. Balandat, E. Bakshy, J. Doppa and David Eriksson. "Bayesian Optimization over High-Dimensional Combinatorial Spaces via Dictionary-based Embeddings." \*\* (2023): 7021-7039. [https://doi.org/10.48550/arxiv.2303.01774](https://doi.org/10.48550/arxiv.2303.01774)

Vu Viet Hoang, H. Tran, Sunil Gupta and Vu Nguyen. "High Dimensional Bayesian Optimization using Lasso Variable Selection." *ArXiv*, abs/2504.01743 (2025). [https://doi.org/10.48550/arxiv.2504.01743](https://doi.org/10.48550/arxiv.2504.01743)

Saeed Amiri-Chimeh, Hassan Haghighi and M. Vahidi-Asl. "JAHAN: A framework for procedural generation of game maps from design specifications." *Entertain. Comput.*, 50 (2024): 100644\. [https://doi.org/10.1016/j.entcom.2024.100644](https://doi.org/10.1016/j.entcom.2024.100644)

Ling Jin and Dawei Zhan. "An Adaptive Line Search Method for High-Dimensional Bayesian Optimization." *2025 7th International Conference on Data-driven Optimization of Complex Systems (DOCS)* (2025): 80-85. [https://doi.org/10.1109/docs67533.2025.11200541](https://doi.org/10.1109/docs67533.2025.11200541)

J. Togelius, M. Preuss and Georgios N. Yannakakis. "Towards multiobjective procedural map generation." \*\* (2010): 3:1-3:8. [https://doi.org/10.1145/1814256.1814259](https://doi.org/10.1145/1814256.1814259)

Leonard Papenmeier, Matthias Poloczek and Luigi Nardi. "Understanding High-Dimensional Bayesian Optimization." *ArXiv*, abs/2502.09198 (2025). [https://doi.org/10.48550/arxiv.2502.09198](https://doi.org/10.48550/arxiv.2502.09198)

Sana Alyaseri. "Evaluating Alternative Metaheuristic Algorithms for Procedural Content Generation in Game Design." *Rangahau Aranga: AUT Graduate Review* (2023). [https://doi.org/10.24135/rangahau-aranga.v2i3.181](https://doi.org/10.24135/rangahau-aranga.v2i3.181)

Xingchen Wan, Vu Nguyen, Huong Ha, Binxin Ru, Cong Lu and Michael A. Osborne. "Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces." \*\* (2021): 10663-10674.

Noor Shaker, Julian Togelius and Mark J. Nelson. "Procedural Content Generation in Games." \*\* (2016): 1-237. [https://doi.org/10.1007/978-3-319-42716-4](https://doi.org/10.1007/978-3-319-42716-4)

H. Bian, Jie Tian, Jialiang Yu and Han Yu. "Bayesian Co-evolutionary Optimization based entropy search for high-dimensional many-objective optimization." *Knowl. Based Syst.*, 274 (2023): 110630\. [https://doi.org/10.1016/j.knosys.2023.110630](https://doi.org/10.1016/j.knosys.2023.110630)

Daniele F. Silva, R. Torchelsen and Marilton S. Aguiar. "Procedural game level generation with GANs: potential, weaknesses, and unresolved challenges in the literature." *Multimedia Tools and Applications*, 84 (2025): 34179 \- 34205\. [https://doi.org/10.1007/s11042-025-20612-9](https://doi.org/10.1007/s11042-025-20612-9)

Sam Daulton, David Eriksson, M. Balandat and E. Bakshy. "Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces." *ArXiv*, abs/2109.10964 (2021).

\---

# **Evidence suggests distance-weighted gravity models are generally effective proxies for resource accessibility in discrete spatial networks, but their validity is context- and parameter-dependent.**

**Figure 1:** Overall research support for gravity-based accessibility

## **Effectiveness as an Accessibility Proxy**

Gravity-based, distance-decay models are widely used to measure potential access to services (healthcare, parks, jobs, postal services) and are considered more realistic than simple distance or provider-to-population ratios because they jointly represent **supply, demand, and distance deterrence** (Stacherl and Sauzet 2023; Yang et al. 2006, 23-32; Wu et al. 2017, 45-54; Schuurman et al. 2010, 29-45).

Systematic reviews in healthcare show gravity-type and 2SFCA-family models dominate methodological and applied work, largely because they provide interpretable, spatially continuous proxies of access that align with planning needs in discrete road or transit networks (Stacherl and Sauzet 2023; Sun et al. 2024; Ouma et al. 2021; Schuurman et al. 2010, 29-45).

Empirical validations report moderate but meaningful correspondence to observed utilization or flows (e.g., R² ≈ 0.25 against clinic visits; acceptable MAPE) when travel times and capacities are modeled over real road networks (Begicheva and Begicheva 2025; Piovani et al. 2018; Mishina et al. 2024). Gravity-based accessibility to jobs and parks also yields plausible spatial patterns and useful policy signals in urban case studies (Piovani et al. 2018; Hu and Downs 2019; Chen et al. 2023, 904 \- 922).

## **Key Conditions and Limitations**

| Issue | Insight for discrete networks | Citations |
| ----- | ----- | ----- |
| Distance metric | Using **network travel time** (not Euclidean) improves realism (Begicheva and Begicheva 2025; Piovani et al. 2018; Xia et al. 2018; Sun et al. 2024\) | (Begicheva and Begicheva 2025; Piovani et al. 2018; Sun et al. 2024; Xia et al. 2018\) |
| Distance decay | Choice of decay function/parameters strongly affects results and must be calibrated where possible (Begicheva and Begicheva 2025; Sun et al. 2024; Mishina et al. 2024; Kapatsila et al. 2023\) | (Begicheva and Begicheva 2025; Sun et al. 2024; Mishina et al. 2024; Kapatsila et al. 2023\) |
| Competing demand & capacity | Models that include facility capacity and competition better proxy actual accessibility (Begicheva and Begicheva 2025; Sun et al. 2024; Yang et al. 2006, 23-32; Mishina et al. 2024; Schuurman et al. 2010, 29-45) | (Begicheva and Begicheva 2025; Sun et al. 2024; Yang et al. 2006, 23-32; Mishina et al. 2024; Schuurman et al. 2010, 29-45) |
| Scale & zoning | Results are sensitive to zone size/configuration; fine grids or link-based measures mitigate this (Piovani et al. 2018; Hu and Downs 2019; Cooper and Chiaradia 2020, 100525; Kapatsila et al. 2023\) | (Piovani et al. 2018; Hu and Downs 2019; Cooper and Chiaradia 2020, 100525; Kapatsila et al. 2023\) |

**Figure 2:** Main modeling choices shaping gravity performance

## **Conclusion**

Overall, distance-weighted gravity models are widely accepted and empirically useful proxies for resource accessibility on discrete spatial networks, provided that network-based distances, sensible decay functions, and supply–demand constraints are carefully specified and, when possible, calibrated to observed behavior.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

B. Stacherl and Odile Sauzet. "Gravity models for potential spatial healthcare access measurement: a systematic methodological review." *International Journal of Health Geographics*, 22 (2023). [https://doi.org/10.1186/s12942-023-00358-z](https://doi.org/10.1186/s12942-023-00358-z)

S. Begicheva and A. Begicheva. "Modified gravity model for assessing healthcare accessibility: problem statement, algorithm, and implementation." *Journal Of Applied Informatics* (2025). [https://doi.org/10.37791/2687-0649-2025-20-5-4-21](https://doi.org/10.37791/2687-0649-2025-20-5-4-21)

Duccio Piovani, E. Arcaute, Gabriela Uchoa, A. Wilson and M. Batty. "Measuring accessibility using gravity and radiation models." *Royal Society Open Science*, 5 (2018). [https://doi.org/10.1098/rsos.171668](https://doi.org/10.1098/rsos.171668)

Shijie Sun, Qun Sun, Fubing Zhang and Jingzhen Ma. "A Spatial Accessibility Study of Public Hospitals: A Multi-Mode Gravity-Based Two-Step Floating Catchment Area Method." *Applied Sciences* (2024). [https://doi.org/10.3390/app14177713](https://doi.org/10.3390/app14177713)

Nan Xia, Liang Cheng, Song Chen, Xiaoyan Wei, Wenwen Zong and Manchun Li. "Accessibility based on Gravity-Radiation model and Google Maps API: A case study in Australia." *Journal of Transport Geography* (2018). [https://doi.org/10.1016/j.jtrangeo.2018.09.009](https://doi.org/10.1016/j.jtrangeo.2018.09.009)

Duck-Hye Yang, Robert M. Goerge and R. Mullner. "Comparing GIS-Based Methods of Measuring Spatial Accessibility to Health Services." *Journal of Medical Systems*, 30 (2006): 23-32. [https://doi.org/10.1007/s10916-006-7400-5](https://doi.org/10.1007/s10916-006-7400-5)

Chao Wu, X. Ye, Qingyun Du and P. Luo. "Spatial effects of accessibility to parks on housing prices in Shenzhen, China." *Habitat International*, 63 (2017): 45-54. [https://doi.org/10.1016/j.habitatint.2017.03.010](https://doi.org/10.1016/j.habitatint.2017.03.010)

P. Ouma, P. Macharia, E. Okiro and V. Alegana. "Methods of Measuring Spatial Accessibility to Health Care in Uganda." *Practicing Health Geography* (2021). [https://doi.org/10.1007/978-3-030-63471-1\_6](https://doi.org/10.1007/978-3-030-63471-1_6)

Yujie Hu and J. Downs. "Measuring and Visualizing Place-Based Space-Time Job Accessibility." *ArXiv*, abs/2006.00268 (2019). [https://doi.org/10.1016/j.jtrangeo.2018.12.002](https://doi.org/10.1016/j.jtrangeo.2018.12.002)

Pengshan Chen, Wei Wang, Chong Qian, M. Cao and Tianren Yang. "Gravity-based models for evaluating urban park accessibility: Why does localized selection of attractiveness factors and travel modes matter?." *Environment and Planning B: Urban Analytics and City Science*, 51 (2023): 904 \- 922\. [https://doi.org/10.1177/23998083231206168](https://doi.org/10.1177/23998083231206168)

Crispin H. V. Cooper and A. Chiaradia. "sDNA: 3-d spatial network analysis for GIS, CAD, Command Line & Python." *SoftwareX*, 12 (2020): 100525\. [https://doi.org/10.1016/j.softx.2020.100525](https://doi.org/10.1016/j.softx.2020.100525)

Margarita E. Mishina, Sergey Mityagin, Alexander B. Belyi, Alexander A. Khrulkov and Stanislav Sobolevsky. "Towards Urban Accessibility: Modeling Trip Distribution to Assess the Provision of Social Facilities." *Smart Cities* (2024). [https://doi.org/10.3390/smartcities7050106](https://doi.org/10.3390/smartcities7050106)

Bogdan Kapatsila, M. Palacios, Emily Grisé and A. El-geneidy. "Resolving the accessibility dilemma: Comparing cumulative and gravity-based measures of accessibility in eight Canadian cities." *Journal of Transport Geography* (2023). [https://doi.org/10.1016/j.jtrangeo.2023.103530](https://doi.org/10.1016/j.jtrangeo.2023.103530)

N. Schuurman, Myriam Bérubé and Valorie A. Crooks. "Measuring potential spatial access to primary health care physicians using a modified gravity model." *Canadian Geographer*, 54 (2010): 29-45. [https://doi.org/10.1111/j.1541-0064.2009.00301.x](https://doi.org/10.1111/j.1541-0064.2009.00301.x)

\---

# **Evidence suggests combining spatial autocorrelation with inequality indices gives a richer picture of structural inequality, but explicit Moran’s I–JFI hybrids are still rare.**

**Figure 1:** Consensus on combining spatial and inequality measures.

## **How spatial statistics and inequality indices interact**

**1\. Why inequality indices alone are insufficient**

Classic inequality measures (Gini, Theil, Atkinson, JFI-type ratios) are *aspatial*: they ignore where advantaged and disadvantaged groups are located, so very different spatial patterns can yield the same index value (Panzera and Postiglione 2019, 379-394; Rey and Smith 2013, 55-70; Puttanapong et al. 2022). This anonymity property hides clustered deprivation or segregation that is central to structural inequality (Panzera and Postiglione 2019, 379-394; Rey and Smith 2013, 55-70).

Frameworks for urban equity therefore explicitly recommend adding spatial correlation measures (e.g., Moran’s I, local I, Getis–Ord) to inequality indices to capture both magnitude and spatial pattern of inequality (Puttanapong et al. 2022; Dong et al. 2023; Clark et al. 2022, 145 \- 163; Zhao et al. 2024).

**2\. Evidence from combined or integrated approaches**

* A spatial decomposition of the Gini coefficient partitions inequality into components between spatial neighbors vs non‑neighbors, explicitly embedding spatial autocorrelation into an inequality index; this enhances detection of spatially-structured disparities compared with Gini alone (Rey and Smith 2013, 55-70).  
* A Gini–correlation–based measure that jointly accounts for inequality and spatial autocorrelation can identify how much of overall inequality is due to spatial patterning, revealing regional contributions that standard indices miss (Panzera and Postiglione 2019, 379-394).  
* Health‑resource inequality in Shanghai was assessed using Gini/Theil plus global and local Moran’s I; Moran’s I highlighted persistent high–high and low–low clusters despite declining global inequality indices, pinpointing structurally underserved areas (Dong et al. 2023).  
* Similar integrations of spatial clustering measures with inequality indicators in Thailand and multi‑sector urban equity data frameworks show that adding spatial statistics uncovers development cores and peripheries not visible from inequality indices alone (Puttanapong et al. 2022; Clark et al. 2022, 145 \- 163).

## **Relation to distributive fairness metrics (JFI, ML fairness)**

Location-based ML fairness work treats space itself as a protected attribute and optimizes prediction under spatial fairness criteria, effectively combining spatial structure with fairness metrics to reduce disparities across locations (He et al. 2024, 4750-4765; Xie et al. 2022, 12208-12216; Yan and Howe 2020, 1079-1087). While these use custom fairness gaps rather than JFI, they support the principle that **joint spatial–fairness formulations improve detection and mitigation of structurally biased patterns** (He et al. 2024, 4750-4765; Xie et al. 2022, 12208-12216; Yan and Howe 2020, 1079-1087).

### **Summary table**

| Aspect captured | Inequality/JFI only | With Moran’s I / spatial terms | Citations |
| ----- | ----- | ----- | ----- |
| Overall disparity magnitude | Yes | Yes | (Panzera and Postiglione 2019, 379-394; Rey and Smith 2013, 55-70; Puttanapong et al. 2022; Dong et al. 2023\) |
| Spatial clustering of advantage/disadvantage | No | Yes (global & local) | (Panzera and Postiglione 2019, 379-394; Rey and Smith 2013, 55-70; Puttanapong et al. 2022; Dong et al. 2023; Clark et al. 2022, 145 \- 163; Zhao et al. 2024\) |
| Identification of structural hot/cold spots | Weak | Strong (clusters, neighbor vs non‑neighbor) | (Rey and Smith 2013, 55-70; Puttanapong et al. 2022; Dong et al. 2023; Clark et al. 2022, 145 \- 163; Perchinunno et al. 2024\) |

**Figure 2:** Complementary roles of inequality and spatial metrics.

## **Conclusion**

There is consistent evidence that pairing inequality/fairness indices with spatial autocorrelation metrics like Moran’s I reveals clustered, structurally patterned inequities that scalar fairness measures alone can miss. Direct Moran’s I–JFI hybrids are not yet standard, but conceptually and empirically, integrated spatial–fairness approaches offer more sensitive detection of structural inequality than either family of metrics in isolation.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Domenica Panzera and Paolo Postiglione. "Measuring the Spatial Dimension of Regional Inequality: An Approach Based on the Gini Correlation Measure." *Social Indicators Research*, 148 (2019): 379-394. [https://doi.org/10.1007/s11205-019-02208-7](https://doi.org/10.1007/s11205-019-02208-7)

S. Rey and Richard J. Smith. "A spatial decomposition of the Gini coefficient." *Letters in Spatial and Resource Sciences*, 6 (2013): 55-70. [https://doi.org/10.1007/s12076-012-0086-z](https://doi.org/10.1007/s12076-012-0086-z)

Nattapong Puttanapong, A. Luenam and Pit Jongwattanakul. "Spatial Analysis of Inequality in Thailand: Applications of Satellite Data and Spatial Statistics/Econometrics." *Sustainability* (2022). [https://doi.org/10.3390/su14073946](https://doi.org/10.3390/su14073946)

Enhong Dong, Xiaoting Sun, Ting Xu, Shixiang Zhang, Tao Wang, Lufa Zhang and Weimin Gao. "Measuring the inequalities in healthcare resource in facility and workforce: A longitudinal study in China." *Frontiers in Public Health*, 11 (2023). [https://doi.org/10.3389/fpubh.2023.1074417](https://doi.org/10.3389/fpubh.2023.1074417)

Erhu He, Yiqun Xie, Weiye Chen, Sergii Skakun, Han Bao, Rahul Ghosh, Praveen Ravirathinam and Xiaowei Jia. "Learning With Location-Based Fairness: A Statistically-Robust Framework and Acceleration." *IEEE Transactions on Knowledge and Data Engineering*, 36 (2024): 4750-4765. [https://doi.org/10.1109/tkde.2024.3371460](https://doi.org/10.1109/tkde.2024.3371460)

Yiqun Xie, Erhu He, X. Jia, Weiye Chen, S. Skakun, Han Bao, Zhenling Jiang, Rahul Ghosh and Praveen Ravirathinam. "Fairness by "Where": A Statistically-Robust and Model-Agnostic Bi-level Learning Framework." \*\* (2022): 12208-12216. [https://doi.org/10.1609/aaai.v36i11.21481](https://doi.org/10.1609/aaai.v36i11.21481)

Lara P. Clark, Samuel Tabory, Kangkang Tong, Joseph L. Servadio, K. Kappler, C. Xu, A. Lawal, Peter Wiringa, L. Kne, R. Feiock, J. Marshall, A. Russell and A. Ramaswami. "A data framework for assessing social inequality and equity in multi‐sector social, ecological, infrastructural urban systems: Focus on fine‐spatial scales." *Journal of Industrial Ecology*, 26 (2022): 145 \- 163\. [https://doi.org/10.1111/jiec.13222](https://doi.org/10.1111/jiec.13222)

Paola Perchinunno, Samuela L'Abbate, C. Crocetta and L. S. Alaimo. "Multidimensional Statistical Analysis of Social Inequalities in Italy." *Socio-Economic Planning Sciences* (2024). [https://doi.org/10.1016/j.seps.2024.102005](https://doi.org/10.1016/j.seps.2024.102005)

Minjun Zhao, Ning Liu, Jinliu Chen, Danqing Wang, Pengcheng Li, Di Yang and Pu Zhou. "Navigating Post-COVID-19 Social–Spatial Inequity: Unravelling the Nexus between Community Conditions, Social Perception, and Spatial Differentiation." *Land* (2024). [https://doi.org/10.3390/land13040563](https://doi.org/10.3390/land13040563)

An Yan and Bill Howe. "Fairness-Aware Demand Prediction for New Mobility." \*\* (2020): 1079-1087. [https://doi.org/10.1609/aaai.v34i01.5458](https://doi.org/10.1609/aaai.v34i01.5458)

\---

# **Hexagonal vs square grids in spatial autocorrelation: hexagons change neighbor structure, test distributions, and sensitivity, but not the basic idea of Moran’s I.**

## **Key differences for Moran’s I and related measures**

**Neighbor structure and weights**

* Hexagonal tessellations have **six equidistant neighbors** with isotropic connectivity; squares often use rook (4) or queen (8) neighbors with anisotropic directions and unequal distances on a sphere (Boots and Tiefelsdorf 2000, 319-348; Wang et al. 2024, 299 \- 313; Magillo 2025, 100695).  
* Different tessellations and adjacency schemes generate **different distributions of Moran’s I and local Ii**, even for the same underlying pattern; normal approximations are not always valid and differ across triangles, squares, and hexagons (Boots and Tiefelsdorf 2000, 319-348).  
* In simultaneous autoregressive (SAR) models, the sampling distribution and variance of the spatial parameter ρ differ between square (rook/queen) and hexagonal (rook) cases, requiring configuration‑specific variance functions for hypothesis tests (Luo et al. 2018, 476).

### **Overview of impacts**

| Aspect | Squares (rook/queen) | Hexagons | Citations |
| ----- | ----- | ----- | ----- |
| Neighbors per cell | 4 or 8, mixed distances/directions | 6, equal distances, isotropic | (Boots and Tiefelsdorf 2000, 319-348; Luo et al. 2018, 476; Wang et al. 2024, 299 \- 313; Magillo 2025, 100695\) |
| Moran’s I distribution | Varies by scheme; normality sometimes poor | Different moments; normality feasibility differs | (Boots and Tiefelsdorf 2000, 319-348; Luo et al. 2018, 476; Rodrigues and Tenedório 2016, 65-77) |
| Edge effects / contiguity artifacts | Stronger corner/edge effects in some tasks | Fewer corner artifacts, more uniform adjacency | (Boots and Tiefelsdorf 2000, 319-348; Wang et al. 2024, 299 \- 313; Monzur et al. 2024\) |
| Matching irregular areal data | Squares align with rasters; less like real polygons | Hexagons better proxy for vector maps | (Boots and Tiefelsdorf 2000, 319-348; Rodrigues and Tenedório 2016, 65-77; Magillo 2025, 100695\) |

**Figure 1:** Comparison of how grid shape alters neighbors and Moran’s I behavior.

## **Broader implications for spatial pattern analysis**

* Hexagonal grids tend to give **more consistent spatial metrics** (e.g., for urban CA models and isoline extraction) due to isotropy and uniform neighborhoods (Nugraha et al. 2020, 845 \- 860; Wang et al. 2024, 299 \- 313).  
* When global Moran’s I is applied to empirical population data, switching between administrative units and hexagonal tessellations changes the **magnitude and even presence of detected autocorrelation**, reflecting scale and zoning effects (Rodrigues and Tenedório 2016, 65-77).  
* For other correlation measures (e.g., pair correlation functions), explicit formulations differ by tessellation, but the core idea is preserved; hexagonal lattices simply require distance/neighbor definitions consistent with their geometry (Gavagnin et al. 2018, 062104 ).

## **Conclusion**

Hexagonal grids do not change the conceptual definition of spatial autocorrelation, but their more isotropic neighbor structure alters the **weights matrix, sampling distributions, edge effects, and sometimes the detected strength and significance** of autocorrelation compared with square rasters.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

B. Boots and M. Tiefelsdorf. "Global and local spatial autocorrelation in bounded regular tessellations." *Journal of Geographical Systems*, 2 (2000): 319-348. [https://doi.org/10.1007/pl00011461](https://doi.org/10.1007/pl00011461)

Qing Luo, D. Griffith and Huayi Wu. "On the Statistical Distribution of the Nonzero Spatial Autocorrelation Parameter in a Simultaneous Autoregressive Model." *ISPRS Int. J. Geo Inf.*, 7 (2018): 476\. [https://doi.org/10.3390/ijgi7120476](https://doi.org/10.3390/ijgi7120476)

Aditya Tafta Nugraha, B. Waterson, S. Blainey and Frederick J. Nash. "On the consistency of urban cellular automata models based on hexagonal and square cells." *Environment and Planning B: Urban Analytics and City Science*, 48 (2020): 845 \- 860\. [https://doi.org/10.1177/2399808319898501](https://doi.org/10.1177/2399808319898501)

Wenbo Wang, Liangchen Zhou, A-Xing Zhu and Guonian Lv. "Isoline extraction based on a global hexagonal grid." *Cartography and Geographic Information Science*, 52 (2024): 299 \- 313\. [https://doi.org/10.1080/15230406.2024.2359709](https://doi.org/10.1080/15230406.2024.2359709)

A. M. Rodrigues and J. Tenedório. "Sensitivity Analysis of Spatial Autocorrelation Using Distinct Geometrical Settings: Guidelines for the Quantitative Geographer." *Int. J. Agric. Environ. Inf. Syst.*, 7 (2016): 65-77. [https://doi.org/10.4018/ijaeis.2016010105](https://doi.org/10.4018/ijaeis.2016010105)

Enrico Gavagnin, Jennifer P. Owen and Christian A. Yates. "Pair correlation functions for identifying spatial correlation in discrete domains.." *Physical review. E*, 97 6-1 (2018): 062104\. [https://doi.org/10.1103/physreve.97.062104](https://doi.org/10.1103/physreve.97.062104)

Tawhid Monzur, Tanzila Tabassum and Nawshin Bashir. "Alternative Tessellations for the Identification of Urban Employment Subcenters: A Comparison of Triangles, Squares, and Hexagons." *Journal of Geovisualization and Spatial Analysis*, 8 (2024). [https://doi.org/10.1007/s41651-024-00200-5](https://doi.org/10.1007/s41651-024-00200-5)

Paola Magillo. "Non-square grids: A new trend in imaging and modeling?." *Comput. Sci. Rev.*, 56 (2025): 100695\. [https://doi.org/10.1016/j.cosrev.2024.100695](https://doi.org/10.1016/j.cosrev.2024.100695)

