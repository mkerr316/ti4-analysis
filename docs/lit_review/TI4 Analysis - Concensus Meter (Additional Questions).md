## **Jain’s Index and Spatial Grids: What It Captures and What It Misses**

**Short answer:** Jain’s Fairness Index **reliably measures how equal numerical allocations are across grid nodes**, but **on its own it does not fully capture spatial fairness** (who is disadvantaged where in the grid).

**Figure 1:** Consensus on Jain index as fairness measure

## **What Jain’s Fairness Index Does Well**

* JFI is axiomatic, bounded, and **Schur‑concave**, making it a robust scalar measure of how evenly a resource vector (e.g., curtailment, prices, comfort) is shared across agents or grid nodes (Lan et al. 2009, 1-9).  
* In power and energy grids, it is widely used to **penalize unequal treatment**: in PV curtailment schemes, topology reconfiguration, harmonic mitigation, EV charging coordination, and transactive energy markets, higher JFI aligns with more balanced burdens or benefits across participants (Gupta and Molzahn 2024; Zarabie et al. 2018, 6029-6040; Kotsonias et al. 2024, 1-5; Kotsonias et al. 2025, 1273-1286; Zhou et al. 2025; AbdulSamadElSkaff et al. 2025, 1-6).  
* Optimization formulations that embed JFI (often via second‑order cone reformulations) can systematically trade efficiency against an explicit notion of numerical fairness (Rezaeinia et al. 2023, 1-23; Kotsonias et al. 2024, 1-5; Kotsonias et al. 2025, 1273-1286).

## **Key Limitations in Spatial Fairness**

* JFI evaluates **only the distribution of quantities**, not their **geographic arrangement**: two grids with identical per‑node allocations but different spatial clustering of disadvantage get the same JFI (Heylen et al. 2019).  
* Power‑system fairness work therefore often turns to **Gini/variance‑based indices** or α‑fairness families that better characterize inequality patterns and equity trade‑offs in reliability or access across locations (Heylen et al. 2019; Blanco and G'azquez 2022, 106287).  
* Multi‑metric extensions, such as **multi‑Jain indices** over several per‑entity features (proportionality, envy‑freeness, equity), show that a single scalar JFI is insufficient to represent richer fairness notions even before adding spatial structure (Köppen et al. 2013, 841-846).  
* Dedicated models of **spatial data fairness** treat location as a protected attribute and define fairness via regional outcome distributions, not via Jain alone (Sacharidis et al. 2023, 485-491; Shaham et al. 2022, 167 \- 179).

### **Summary Table: JFI vs. Spatial Fairness Needs**

| Aspect | JFI Captures? | Notes | Citations |
| ----- | ----- | ----- | ----- |
| Evenness across nodes | Yes (strong) | Reliable scalar equality metric | (Zhou et al. 2025; Gupta and Molzahn 2024; Zarabie et al. 2018, 6029-6040; Kotsonias et al. 2024, 1-5; Kotsonias et al. 2025, 1273-1286; Lan et al. 2009, 1-9) |
| Spatial pattern / clustering | No | Ignores adjacency, regions | (Sacharidis et al. 2023, 485-491; Shaham et al. 2022, 167 \- 179; Heylen et al. 2019; Blanco and G'azquez 2022, 106287\) |
| Multi-dimensional justice | Partly (via variants) | Needs multi-feature or extra indices | (Heylen et al. 2019; Köppen et al. 2013, 841-846; Blanco and G'azquez 2022, 106287\) |

**Figure 2:** Comparison of Jain index and spatial fairness needs

## **Conclusion**

Jain’s Fairness Index is a **reliable measure of numerical equality** across grid locations, and is useful inside optimization for “not over‑burdening” any single node. But **fairness in spatial grids typically requires additional, spatially aware metrics** (e.g., Gini, reliability inequity indices, spatial‑fairness tests) to capture where and how inequity is distributed.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Lixia Zhou, Dawei Huo, Jian Chen, Bo Bo and Hao Li. "Federated reinforcement learning with constrained markov decision processes and graph neural networks for fair and grid-constrained coordination of large-scale electric vehicle charging networks." *Scientific Reports*, 15 (2025). [https://doi.org/10.1038/s41598-025-22482-5](https://doi.org/10.1038/s41598-025-22482-5)

Rahul K. Gupta and Daniel K. Molzahn. "Improving Fairness in Photovoltaic Curtailments via Daily Topology Reconfiguration for Voltage Control in Power Distribution Networks." *ArXiv*, abs/2403.07853 (2024). [https://doi.org/10.1016/j.apenergy.2025.126543](https://doi.org/10.1016/j.apenergy.2025.126543)

A. K. Zarabie, Sanjoy Das and M. N. Faqiry. "Fairness-Regularized DLMP-Based Bilevel Transactive Energy Mechanism in Distribution Systems." *IEEE Transactions on Smart Grid*, 10 (2018): 6029-6040. [https://doi.org/10.1109/tsg.2019.2895527](https://doi.org/10.1109/tsg.2019.2895527)

Nahid Rezaeinia, J. Goez and Mario Guajardo. "On efficiency and the Jain’s fairness index in integer assignment problems." *Computational Management Science*, 20 (2023): 1-23. [https://doi.org/10.1007/s10287-023-00477-9](https://doi.org/10.1007/s10287-023-00477-9)

Andreas Kotsonias, L. Hadjidemetriou, M. Asprou and C. Panayiotou. "Increasing the Fairness of Photovoltaic Curtailments for Voltage Management in Distribution Grids." *2024 IEEE PES Innovative Smart Grid Technologies Europe (ISGT EUROPE)* (2024): 1-5. [https://doi.org/10.1109/isgteurope62998.2024.10863114](https://doi.org/10.1109/isgteurope62998.2024.10863114)

Dimitris Sacharidis, G. Giannopoulos, George Papastefanatos and Kostas Stefanidis. "Auditing for Spatial Fairness." \*\* (2023): 485-491. [https://doi.org/10.48550/arxiv.2302.12333](https://doi.org/10.48550/arxiv.2302.12333)

Sina Shaham, Gabriel Ghinita and C. Shahabi. "Models and Mechanisms for Spatial Data Fairness." *Proceedings of the VLDB Endowment. International Conference on Very Large Data Bases*, 16 (2022): 167 \- 179\. [https://doi.org/10.14778/3565816.3565820](https://doi.org/10.14778/3565816.3565820)

Evelyn Heylen, M. Ovaere, S. Proost, Geert Deconinck and D. Hertem. "Fairness and inequality in power system reliability: Summarizing indices." *Electric Power Systems Research* (2019). [https://doi.org/10.1016/j.epsr.2018.11.011](https://doi.org/10.1016/j.epsr.2018.11.011)

M. Köppen, K. Ohnishi and M. Tsuru. "Multi-Jain Fairness Index of Per-Entity Allocation Features for Fair and Efficient Allocation of Network Resources." *2013 5th International Conference on Intelligent Networking and Collaborative Systems* (2013): 841-846. [https://doi.org/10.1109/incos.2013.161](https://doi.org/10.1109/incos.2013.161)

Andreas Kotsonias, Lysandros Tziovani, L. Hadjidemetriou, M. Asprou and Christos G. Panayiotou. "Mitigation of Harmonic Distortions in Low Voltage Distribution Grids With Fair Utilization of Resources." *IEEE Transactions on Smart Grid*, 16 (2025): 1273-1286. [https://doi.org/10.1109/tsg.2024.3460978](https://doi.org/10.1109/tsg.2024.3460978)

Yara AbdulSamadElSkaff, Hugo Joudrier, Y. Gaoua and Quoc Tuan Tran. "Towards a Fair Disaggregation in Hierarchical Cloud – Edge Control of Distributed Local Energy Communities." *2025 IEEE Kiel PowerTech* (2025): 1-6. [https://doi.org/10.1109/powertech59965.2025.11180730](https://doi.org/10.1109/powertech59965.2025.11180730)

V. Blanco and Ricardo G'azquez. "Fairness in maximal covering location problems." *Comput. Oper. Res.*, 157 (2022): 106287\. [https://doi.org/10.1016/j.cor.2023.106287](https://doi.org/10.1016/j.cor.2023.106287)

T. Lan, D. Kao, M. Chiang and A. Sabharwal. "An Axiomatic Theory of Fairness in Network Resource Allocation." *2010 Proceedings IEEE INFOCOM* (2009): 1-9. [https://doi.org/10.1109/infcom.2010.5461911](https://doi.org/10.1109/infcom.2010.5461911)

\---

# **Yes, fairness measurement is considered crucial in multi-agent resource allocation, alongside efficiency and satisfaction.**

**Figure 1:** Consensus that fairness is a key design objective

## **Why Fairness Measurement Matters**

* **Core design objective, not an add‑on**: Many works treat fairness as a fundamental criterion next to efficiency, explicitly modeling resource allocation as a joint fairness–efficiency problem (Salem et al. 2022, 1 \- 36; Guo et al. 2023; Nicosia et al. 2015; Li 2025).  
* **Social acceptability & human trust**: Fair allocations are needed for social acceptance of algorithms in human–agent and human–AI teams; perceived unfairness degrades trust and willingness to use the system (Chang et al. 2025; Hosseini et al. 2025).  
* **Nonprofit and public-good contexts**: In nonprofit operations and social services, explicitly measuring fairness (e.g., via demand shortfall) is key to ensure vulnerable communities are not systematically under-served, even when some efficiency is sacrificed (Yuanzheng et al. 2023, 1778 \- 1792).  
* **Algorithm design & tuning**: Modern frameworks (e.g., DECAF, GIFF, Fair-PPO) embed explicit fairness metrics into reinforcement learning or optimization objectives to trade off utility and equity in ridesharing, homelessness prevention, hospital scheduling, and caching (Kumar and Yeoh 2025; Kumar and Yeoh 2025, 2591-2593; Salem et al. 2022, 1 \- 36; Malfa et al. 2024; Malfa et al. 2025).

## **Trade-offs and Insights from Measuring Fairness**

* **Price of fairness**: Research quantifies how much system efficiency is lost when fairness constraints are imposed, helping designers choose appropriate compromise points (Sun and Li 2024, 1800-1808; Nicosia et al. 2015; Li 2025; Yuanzheng et al. 2023, 1778 \- 1792).  
* **Different notions capture different values**: Studies compare equality, need-based, capability-based, and group-based notions (envy-freeness, proportionality, protected-attribute fairness), showing that the “right” metric depends on the application (Gharbi et al. 2025; Guo et al. 2023; Ekpo et al. 2025; Hosseini et al. 2025; Bu et al. 2023, 77-94; Malfa et al. 2024).

### **Example Perspectives**

| Role / Setting | Why fairness measurement is important | Citations |
| ----- | ----- | ----- |
| Self-governing MAS | Agents evaluate fairness to balance needs vs equity | (Gharbi et al. 2025\) |
| Dynamic online allocation (caching, cloud) | Guarantees group fairness under arrivals over time | (Zargari et al. 2025, 1 \- 51; Salem et al. 2022, 1 \- 36\) |
| Human–agent teams, healthcare, bandits | Aligns allocations with human notions of fair share | (Ekpo et al. 2025; Chang et al. 2025; Chen et al. 2019, 181-190) |

**Figure 2:** Contexts where explicit fairness metrics are critical

## **Conclusion**

Across theory and applications, fairness metrics are treated as essential in multi-agent resource allocation: they guide algorithm design, expose efficiency–equity trade-offs, and are necessary for social and ethical acceptability, especially where agents or people differ in needs, capabilities, or protected attributes.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Atef Gharbi, Mohamed Ayari, Nasser Albalawi, Yamen El Touati and Zeineb Klai. "A Comparative Analysis of Fairness and Satisfaction in Multi-Agent Resource Allocation: Integrating Borda Count and K-Means Approaches with Distributive Justice Principles." *Mathematics* (2025). [https://doi.org/10.3390/math13152355](https://doi.org/10.3390/math13152355)

Ashwin Kumar and William Yeoh. "A General Incentives-Based Framework for Fairness in Multi-agent Resource Allocation." *ArXiv*, abs/2510.26740 (2025). [https://doi.org/10.48550/arxiv.2510.26740](https://doi.org/10.48550/arxiv.2510.26740)

Faraz Zargari, Hossein Nekouyan, Bo Sun and Xiaoqi Tan. "Online Allocation with Multi-Class Arrivals: Group Fairness vs Individual Welfare." *Proceedings of the ACM on Measurement and Analysis of Computing Systems*, 9 (2025): 1 \- 51\. [https://doi.org/10.1145/3727120](https://doi.org/10.1145/3727120)

Ashwin Kumar and William Yeoh. "DECAF: Learning to be Fair in Multi-agent Resource Allocation." \*\* (2025): 2591-2593. [https://doi.org/10.48550/arxiv.2502.04281](https://doi.org/10.48550/arxiv.2502.04281)

T. Si Salem, G. Iosifidis and G. Neglia. "Enabling Long-term Fairness in Dynamic Resource Allocation." *Proceedings of the ACM on Measurement and Analysis of Computing Systems*, 6 (2022): 1 \- 36\. [https://doi.org/10.1145/3570606](https://doi.org/10.1145/3570606)

Ankang Sun and Bo Li. "Allocating Contiguous Blocks of Indivisible Chores Fairly: Revisited." \*\* (2024): 1800-1808. [https://doi.org/10.5555/3635637.3663042](https://doi.org/10.5555/3635637.3663042)

Hao Guo, Weidong Li and Bin Deng. "A Survey on Fair Allocation of Chores." *Mathematics* (2023). [https://doi.org/10.3390/math11163616](https://doi.org/10.3390/math11163616)

Promise Ekpo, Brian La, Thomas Wiener, Saesha Agarwal, Arshia Agrawal, Gonzalo Gonzalez-Pumariega, Lekan P. Molu and Angelique Taylor. "Skill-Aligned Fairness in Multi-Agent Learning for Collaboration in Healthcare." *ArXiv*, abs/2508.18708 (2025). [https://doi.org/10.48550/arxiv.2508.18708](https://doi.org/10.48550/arxiv.2508.18708)

G. Nicosia, A. Pacifici and U. Pferschy. "Price of Fairness for allocating a bounded resource." *ArXiv*, abs/1508.05253 (2015). [https://doi.org/10.1016/j.ejor.2016.08.013](https://doi.org/10.1016/j.ejor.2016.08.013)

M. L. Chang, Kim Baraka, Greg Trafton, Zach Lalu Vazhekatt and Andrea Lockerd Thomaz. "Fairness and Efficiency in Human-Agent Teams: An Iterative Algorithm Design Approach." *ArXiv*, abs/2505.16171 (2025). [https://doi.org/10.48550/arxiv.2505.16171](https://doi.org/10.48550/arxiv.2505.16171)

Hadi Hosseini, Joshua Kavner, Samarth Khanna, Sujoy Sikdar and Lirong Xia. "Bridging Theory and Perception in Fair Division: A Study on Comparative and Fair Share Notions." *ArXiv*, abs/2505.10433 (2025). [https://doi.org/10.48550/arxiv.2505.10433](https://doi.org/10.48550/arxiv.2505.10433)

Zihao Li. "Fairness and efficiency in resource allocation." \*\* (2025). [https://doi.org/10.32657/10356/182150](https://doi.org/10.32657/10356/182150)

Yuanzheng Ma, Tong-fei Wang and Huan Zheng. "On fairness and efficiency in nonprofit operations: Dynamic resource allocations." *Production and Operations Management*, 32 (2023): 1778 \- 1792\. [https://doi.org/10.1111/poms.13940](https://doi.org/10.1111/poms.13940)

Xiaolin Bu, Zihao Li, Shengxin Liu, Jiaxin Song and Biaoshuai Tao. "Fair Division with Allocator's Preference." \*\* (2023): 77-94. [https://doi.org/10.48550/arxiv.2310.03475](https://doi.org/10.48550/arxiv.2310.03475)

G. Malfa, Jie M. Zhang, Michael Luck and Elizabeth Black. "Using Protected Attributes to Consider Fairness in Multi-Agent Systems." *ArXiv*, abs/2410.12889 (2024). [https://doi.org/10.48550/arxiv.2410.12889](https://doi.org/10.48550/arxiv.2410.12889)

G. Malfa, Jie M. Zhang, Michael Luck and Elizabeth Black. "Fairness Aware Reinforcement Learning via Proximal Policy Optimization." *ArXiv*, abs/2502.03953 (2025). [https://doi.org/10.48550/arxiv.2502.03953](https://doi.org/10.48550/arxiv.2502.03953)

Yifang Chen, Alex Cuellar, Haipeng Luo, Jignesh Modi, Heramb Nemlekar and S. Nikolaidis. "Fair Contextual Multi-Armed Bandits: Theory and Experiments." \*\* (2019): 181-190.

\---

**Evidence is indirect and does not show a clear link between spatial autocorrelation and player fairness perceptions in games**

**Figure 1:** Consensus on spatial structure–fairness perception link

## **What is known**

**Spatial autocorrelation & fairness (technical level, not perception)**  
 Work in algorithmic fairness with spatial data shows that **spatial autocorrelation can confound fairness/bias metrics**, and proposes statistical methods to detect and filter it before evaluating fairness (Flynn et al. 2021; Majumdar et al. 2021, 6-18). These studies demonstrate that spatial dependence matters for *measuring* fairness in spatial systems, but they do not involve human subjects or perceived fairness in games.

**Spatial structure & evolution of fair strategies (not perceptions)**  
 In spatial versions of the ultimatum game, local spatial interaction can **promote the evolution of fairer strategies** compared to well-mixed populations (Szolnoki et al. 2012). This shows spatial structure affects which fairness norms emerge, but it models payoff dynamics, not subjective judgments of fairness by players.

**Fairness perception in games and HCI**  
 Research on fairness in games and human–computer/robot interaction focuses on:

* Social rank, identity, or territorial cues in ultimatum-style games shaping what offers are *perceived* as fair (Su et al. 2025; Hanmo et al. 2025)- Fair vs unfair offers affecting **trust in human–computer and human–AI interactions** (Chen et al. 2025; Shin and Lee 2025; Starke et al. 2021)- Perceived fairness around monetization (“pay to win”) in competitive online games (Freeman et al. 2022, 1 \- 24)- Dynamic fairness perceptions toward a robot partner in a multiplayer game setting (Claure et al. 2024)These studies manipulate **social, economic, or procedural variables**, not the spatial autocorrelation of the game space.

**Level balance & skill, without explicit spatial statistics**  
 Game AI work on balancing multiplayer maps across skill levels uses learned models from level layouts to predict win-rate balance, connecting layout to *outcome balance* rather than directly to perceived fairness, and does not use Moran’s I or similar spatial autocorrelation metrics (Stephens and Exton 2022, 827-833).

### **Summary Table**

| Dimension | What varies fairness perception | Role of spatial autocorrelation | Citations |
| ----- | ----- | ----- | ----- |
| Social / identity cues | Rank, group, territorial identity | No explicit spatial statistics | (Su et al. 2025; Hanmo et al. 2025; Cram et al. 2018\) |
| Offers / rewards / monetization | Fair vs unfair splits, in‑game purchases | Not modeled spatially | (Chen et al. 2025; Freeman et al. 2022, 1 \- 24; Shin and Lee 2025; Starke et al. 2021\) |
| Spatial data & fairness metrics | Bias detection on spatial data | Spatial autocorrelation as *statistical confounder* | (Flynn et al. 2021; Majumdar et al. 2021, 6-18) |
| Spatial games (theoretical) | Local interaction structure → fair strategies | No link to human perceptions | (Alonso-Sanz 2017; Szolnoki et al. 2012\) |

**Figure 2:** Where fairness is studied vs. spatial dependence

## **Conclusion**

Existing research shows that **spatial structure can influence which fairness norms emerge and how we statistically assess fairness**, but there is no empirical work directly testing whether *higher or lower spatial autocorrelation in game maps changes players’ perceived fairness*. Any claimed correlation in games remains a plausible but untested hypothesis.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Yaner Su, Ying Zeng, André Aleman, Sander Martens, Pengfei Xu, Yue-jia Luo and K. Goerlich. "Dynamic interplay of hierarchical rank and social contexts in shaping perceived fairness." *NeuroImage*, 317 (2025). [https://doi.org/10.1016/j.neuroimage.2025.121363](https://doi.org/10.1016/j.neuroimage.2025.121363)

Rui Chen, Yating Jin, Lincang Yu, Tobias Tempel, Peng Li, Shi Zhang, Anqi Li and Weijie He. "The Influence of Perceived Fairness on Trust in Human‐Computer Interaction." *International Journal of Psychology*, 60 (2025). [https://doi.org/10.1002/ijop.70111](https://doi.org/10.1002/ijop.70111)

Guo Freeman, K. Wu, Nicholas Nower and D. Y. Wohn. "Pay to Win or Pay to Cheat: How Players of Competitive Online Games Perceive Fairness of In-Game Purchases." *Proceedings of the ACM on Human-Computer Interaction*, 6 (2022): 1 \- 24\. [https://doi.org/10.1145/3549510](https://doi.org/10.1145/3549510)

Cheryl J. Flynn, S. Majumdar and Ritwik Mitra. "Evaluating Fairness in the Presence of Spatial Autocorrelation." *ArXiv*, abs/2101.01703 (2021).

Mincheol Shin and Doohwang Lee. "Human Responses to AI’s Unfair Decisions in the Social Ultimatum Split Game: The Role of Perceived Fairness and Trust in Adoption Intention." *International Journal of Human–Computer Interaction* (2025). [https://doi.org/10.1080/10447318.2025.2567966](https://doi.org/10.1080/10447318.2025.2567966)

Yin Hanmo, Rozainee Khairudin, Shi Yixin, Zhang Xi, Chen Xinyu, Syakirien Yusoff and N. Subhi. "Subjective socioeconomic status moderates depression’s impact on fairness perception in the ultimatum game: A moderated mediation model." *PLOS One*, 20 (2025). [https://doi.org/10.1371/journal.pone.0330870](https://doi.org/10.1371/journal.pone.0330870)

Houston Claure, Kate Candon, Olivia E Clark and Marynel Vázquez. "Multiplayer Space Invaders: A Platform for Studying Evolving Fairness Perceptions in Human-Robot Interaction." *Companion of the 2024 ACM/IEEE International Conference on Human-Robot Interaction* (2024). [https://doi.org/10.1145/3610978.3640687](https://doi.org/10.1145/3610978.3640687)

S. Majumdar, Cheryl J. Flynn and Ritwik Mitra. "Detecting Bias in the Presence of Spatial Autocorrelation." \*\* (2021): 6-18.

C. Starke, Janine Baleis, Birte Keller and Frank Marcinkowski. "Fairness perceptions of algorithmic decision-making: A systematic review of the empirical literature." *Big Data & Society*, 9 (2021). [https://doi.org/10.1177/20539517221115189](https://doi.org/10.1177/20539517221115189)

R. Alonso-Sanz. "Spatial correlated games." *Royal Society Open Science*, 4 (2017). [https://doi.org/10.1098/rsos.171361](https://doi.org/10.1098/rsos.171361)

Laura Cram, Adam B. Moore, Victor M. Olivieri and Felix Suessenbach. "Fair Is Fair, or Is It? Territorial Identity Triggers Influence Ultimatum Game Behavior." *Political Psychology* (2018). [https://doi.org/10.1111/pops.12543](https://doi.org/10.1111/pops.12543)

Conor Stephens and C. Exton. "Balancing Multiplayer Games across Player Skill Levels using Deep Reinforcement Learning." \*\* (2022): 827-833. [https://doi.org/10.5220/0010914200003116](https://doi.org/10.5220/0010914200003116)

A. Szolnoki, M. Perc and G. Szabó. "Accuracy in strategy imitations promotes the evolution of fairness in the spatial ultimatum game." *Europhysics Letters*, 100 (2012). [https://doi.org/10.1209/0295-5075/100/28005](https://doi.org/10.1209/0295-5075/100/28005)

\---

**No, greedy hill-climbing does not guarantee optimal equilibria, even on symmetric maps.**

**Figure 1:** Consensus that greedy local search lacks guarantees.

## **What hill-climbing can (and cannot) guarantee**

Greedy hill-climbing is a **local search** method: from a starting map, it repeatedly moves to a strictly better neighbor until no improvement exists. On rugged, high‑dimensional landscapes—typical for map design and balance—this almost always yields **local optima, not global optima** (Basseur and Goëffon 2015, 688-704; Chinnasamy et al. 2022).

A general review of hill-climbing explicitly notes it is **neither complete nor optimal** in the sense of global optimization; it gets stuck at local maxima, ridges, and plateaus (Chinnasamy et al. 2022). Work on combinatorial fitness landscapes shows that even sophisticated variants (best/first improvement, neutral moves) only improve the quality of local optima, not provide global guarantees (Basseur and Goëffon 2015, 688-704).

Theoretical work on generalized hill-climbing proves convergence to global optima only for special classes of algorithms and under strong conditions (e.g., known global optimum value, specific acceptance rules) (Johnson and Jacobson 2002, 37-57; Johnson and Jacobson 2002, 359-373). These conditions do **not** match standard greedy hill-climbing used in PCG.

## **Evidence from procedural content generation and maps**

In Terra Mystica map generation, steepest-ascent hill climbing with random restarts can find maps that satisfy balance constraints, but there is no claim or proof of global optimality or unique equilibrium (De Araújo et al. 2020, 163 \- 175). Tabu search is introduced precisely to let local search **escape local optima** that simple hill-climbing cannot (Grichshenko et al. 2020).

Search-based PCG over game maps generally relies on **evolutionary or multi-objective algorithms** instead of pure greedy hill-climbing, acknowledging that the landscape is multi-modal and guarantees of optimal equilibrium are unavailable (Volz et al. 2023, 110121; Zafar et al. 2020; Lianbo et al. 2022, 341-354; Bhaumik et al. 2020, 24-30).

Symmetry can be enforced by representation or constraints, but this only restricts the search space; it does not remove local optima, so global equilibrium is still not guaranteed (De Araújo et al. 2020, 163 \- 175; Volz et al. 2023, 110121; Volz et al. 2020, 399-406; Lianbo et al. 2022, 341-354).

## **Conclusion**

Even with symmetric maps and well-designed fitness functions, greedy hill-climbing offers **no theoretical guarantee** of reaching an optimal equilibrium; at best it finds locally balanced maps, often improved via restarts or hybrid/metaheuristic frameworks.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Luiz Jonatã Pires de Araújo, Alexandr Grichshenko, R. Pinheiro, R. D. Saraiva and Susanna Gimaeva. "Map Generation and Balance in the Terra Mystica Board Game Using Particle Swarm and Local Search." *Advances in Swarm Intelligence*, 12145 (2020): 163 \- 175\. [https://doi.org/10.1007/978-3-030-53956-6\_15](https://doi.org/10.1007/978-3-030-53956-6_15)

Alexandr Grichshenko, Luiz Jonatã Pires de Araújo, Susanna Gimaeva and J. A. Brown. "Using Tabu Search Algorithm for Map Generation in the Terra Mystica Tabletop Game." *Proceedings of the 2020 4th International Conference on Intelligent Systems, Metaheuristics & Swarm Intelligence* (2020). [https://doi.org/10.1145/3396474.3396492](https://doi.org/10.1145/3396474.3396492)

Vanessa Volz, B. Naujoks, P. Kerschke and Tea Tušar. "Tools for Landscape Analysis of Optimisation Problems in Procedural Content Generation for Games." *Appl. Soft Comput.*, 136 (2023): 110121\. [https://doi.org/10.1016/j.asoc.2023.110121](https://doi.org/10.1016/j.asoc.2023.110121)

Matthieu Basseur and Adrien Goëffon. "Climbing combinatorial fitness landscapes." *Appl. Soft Comput.*, 30 (2015): 688-704. [https://doi.org/10.1016/j.asoc.2015.01.047](https://doi.org/10.1016/j.asoc.2015.01.047)

Adeel Zafar, Hasan Mujtaba and M. O. Beg. "Search-based procedural content generation for GVG-LG." *Appl. Soft Comput.*, 86 (2020). [https://doi.org/10.1016/j.asoc.2019.105909](https://doi.org/10.1016/j.asoc.2019.105909)

Alan W. Johnson and S. Jacobson. "On the convergence of generalized hill climbing algorithms." *Discret. Appl. Math.*, 119 (2002): 37-57. [https://doi.org/10.1016/s0166-218x(01)00264-5](https://doi.org/10.1016/s0166-218x\(01\)00264-5)

Alan W. Johnson and S. Jacobson. "A class of convergent generalized hill climbing algorithms." *Appl. Math. Comput.*, 125 (2002): 359-373. [https://doi.org/10.1016/s0096-3003(00)00137-5](https://doi.org/10.1016/s0096-3003\(00\)00137-5)

Vanessa Volz, Niels Justesen, Sam Snodgrass, S. Asadi, Sami Purmonen, Christoffer Holmgård, J. Togelius and S. Risi. "Capturing Local and Global Patterns in Procedural Content Generation via Machine Learning." *2020 IEEE Conference on Games (CoG)* (2020): 399-406. [https://doi.org/10.1109/cog47356.2020.9231944](https://doi.org/10.1109/cog47356.2020.9231944)

Sathiyaraj Chinnasamy, M Ramachandran, M. Amudha and Kurinjimalar Ramu. "A Review on Hill Climbing Optimization Methodology." *Recent trends in Management and Commerce* (2022). [https://doi.org/10.46632/rmc/3/1/1](https://doi.org/10.46632/rmc/3/1/1)

Lianbo Ma, Shi Cheng, Mingli Shi and Yi-nan Guo. "Angle-Based Multi-Objective Evolutionary Algorithm Based On Pruning-Power Indicator for Game Map Generation." *IEEE Transactions on Emerging Topics in Computational Intelligence*, 6 (2022): 341-354. [https://doi.org/10.1109/tetci.2021.3067104](https://doi.org/10.1109/tetci.2021.3067104)

Debosmita Bhaumik, A. Khalifa, M. Green and J. Togelius. "Tree Search versus Optimization Approaches for Map Generation." \*\* (2020): 24-30. [https://doi.org/10.1609/aiide.v16i1.7403](https://doi.org/10.1609/aiide.v16i1.7403)

\---

# **Yes, Tabu Search can effectively avoid/escape local optima in map generation-style problems.**

**Figure 1:** Consensus on Tabu Search avoiding local optima.

## **Evidence from Procedural Map Generation**

* A Tabu Search (TS)–based generator for **Terra Mystica** board-game maps explicitly targets the problem of local optima in procedural content generation. TS improved on steepest-ascent hill-climbing by escaping local optima and producing maps that **outperformed existing hand-designed maps** according to game-specific metrics (Grichshenko et al. 2020).  
* Varying the **tabu list size** and **neighborhood size** was important: appropriate settings allowed the search to move away from locally good but globally inferior layouts, validating TS’s feasibility and effectiveness for map generation (Grichshenko et al. 2020).

## **Why Tabu Search Escapes Local Optima**

* TS modifies basic local search by:  
  * Allowing **non-improving moves** when stuck, instead of stopping at a local optimum (Subash et al. 2022).  
  * Using a **tabu list** (short-term memory) to forbid recently visited solutions or moves, preventing cycling back into the same local basin (Subash et al. 2022; Glover 1989, 190-206).  
* More advanced TS memory schemes (e.g., **exponential extrapolation memory**) ensure moves that do not lead back to a specified number of recently visited local optima, systematically pushing the search into new regions of the landscape (Bentsen et al. 2022, 100028; Hanafi et al. 2023, 1037-1055).  
* Tutorials and comparative studies describe TS as a metaheuristic designed specifically to guide local search **“beyond local optimality”** and report strong performance on many NP‑hard layout and routing problems analogous to map/path design (Glover 1990, 74-94; Glover 1989, 190-206; Pirim et al. 2008; Bland 1994, 91-96).

### **Mechanisms & Implications for Map Generation**

| Mechanism | Effect on Local Optima in Maps | Citations |
| ----- | ----- | ----- |
| Tabu list (short-term memory) | Blocks cycling back to recent layouts | (Grichshenko et al. 2020; Subash et al. 2022; Glover 1990, 74-94) |
| Allowing worse moves | Lets search jump out of local optima basins | (Subash et al. 2022; Glover 1990, 74-94; Glover 1989, 190-206) |
| Advanced memory (extrapolation) | Steers away from multiple recent optima regions | (Bentsen et al. 2022, 100028; Hanafi et al. 2023, 1037-1055) |

**Figure 2:** Key Tabu mechanisms that help escape local optima.

## **Conclusion**

Evidence from Terra Mystica map generation plus broader TS research indicates that, with well-chosen neighborhoods and tabu-tenure parameters, Tabu Search is an effective way to avoid or escape local optima in procedural map generation.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Alexandr Grichshenko, Luiz Jonatã Pires de Araújo, Susanna Gimaeva and J. A. Brown. "Using Tabu Search Algorithm for Map Generation in the Terra Mystica Tabletop Game." *Proceedings of the 2020 4th International Conference on Intelligent Systems, Metaheuristics & Swarm Intelligence* (2020). [https://doi.org/10.1145/3396474.3396492](https://doi.org/10.1145/3396474.3396492)

Dr. N. subash, M. Ramachandran, Vimala Saravanan and Vidhya Prasanth. "An Investigation on Tabu Search Algorithms Optimization." *Electrical and Automation Engineering* (2022). [https://doi.org/10.46632/1/1/3](https://doi.org/10.46632/1/1/3)

Håkon Bentsen, Arild Hoff and L. M. Hvattum. "Exponential extrapolation memory for tabu search." *EURO J. Comput. Optim.*, 10 (2022): 100028\. [https://doi.org/10.1016/j.ejco.2022.100028](https://doi.org/10.1016/j.ejco.2022.100028)

S. Hanafi, Yang Wang, F. Glover, Wei-Xing Yang and Rick Hennig. "Tabu search exploiting local optimality in binary optimization." *Eur. J. Oper. Res.*, 308 (2023): 1037-1055. [https://doi.org/10.1016/j.ejor.2023.01.001](https://doi.org/10.1016/j.ejor.2023.01.001)

F. Glover. "Tabu Search: A Tutorial." *Interfaces*, 20 (1990): 74-94. [https://doi.org/10.1287/inte.20.4.74](https://doi.org/10.1287/inte.20.4.74)

F. Glover. "Tabu Search \- Part I." *INFORMS J. Comput.*, 1 (1989): 190-206. [https://doi.org/10.1287/ijoc.1.3.190](https://doi.org/10.1287/ijoc.1.3.190)

Harun Pirim, E. Bayraktar and Burak Eksioglu. "Tabu Search: A Comparative Study." \*\* (2008). [https://doi.org/10.5772/5637](https://doi.org/10.5772/5637)

J. A. Bland. "A derivative-free exploratory tool for function minimisation based on tabu search." *Advances in Engineering Software*, 19 (1994): 91-96. [https://doi.org/10.1016/0965-9978(94)90062-0](https://doi.org/10.1016/0965-9978\(94\)90062-0)

\---

# **While simulated annealing has stronger global‑optimum guarantees, it does not always achieve better global optima in practice than hill-climbing; performance is landscape‑ and tuning‑dependent.**

**Figure 1:** Evidence on SA vs hill-climbing global optimality.

## **Theoretical convergence vs. hill-climbing**

* **Simulated annealing (SA)** allows occasional uphill (worsening) moves, which enables escape from local optima, unlike pure greedy hill‑climbing that only accepts improvements (Henderson et al. 2006, 287-319; Nikolaev and Jacobson 2011).  
* With appropriate (usually very slow, e.g., logarithmic) cooling schedules, classical SA is **provably convergent to the set of global optima** for both discrete and continuous problems (Henderson et al. 2006, 287-319; Alrefaei and Andradóttir 1999, 748-764; Dekkers and Aarts 1991, 367-393; Johnson and Jacobson 2002, 37-57).  
* Generalized hill‑climbing frameworks show that these convergence conditions for SA and broader GHC classes are equivalent under certain assumptions, but standard greedy hill‑climbing lacks such global‑optimality guarantees unless heavily modified or restarted (Jacobson and Yücesan 2004, 387-405; Johnson and Jacobson 2002, 37-57; Johnson and Jacobson 2002, 359-373).

## **Empirical/global‑quality comparisons**

* On benchmark global optimization and econometric problems, SA has repeatedly found **global or near‑global optima** where conventional local methods fail or get trapped in poor local minima (Dekkers and Aarts 1991, 367-393; Goffe et al. 1994, 65-99).  
* In spatially constrained forest planning, simulated annealing consistently produced **better objective values** than random hill‑climbing given equal iterations, and was less sensitive to starting points (Liu et al. 2006, 421-428).  
* In PV maximum‑power‑point tracking under partial shading, a hybrid SA–hill‑climbing controller reached the **true global maximum** (\~600 W) while traditional hill‑climbing settled at a suboptimal local maximum (\~450 W) (Alajmi et al. 2025).

### **When SA does *not* clearly win**

* Finite‑time performance can favor well‑tuned hill‑climbing or restart local search on some landscapes; a key “basins of attraction” parameter determines whether SA or restart‑based methods more efficiently hit the global optimum (Jacobson and Yücesan 2004, 387-405; Orosz and Jacobson 2002, 21-53).  
* Static/high‑temperature SA variants may be **non‑convergent** and behave similarly to heuristic hill‑climbers, trading global guarantees for speed (Orosz and Jacobson 2002, 21-53; Orosz and Jacobson 2002, 165-182).

### **Conclusion**

Overall, simulated annealing is **more likely** than simple hill‑climbing to reach global optima due to its escape mechanism and proven asymptotic convergence, but in finite time it does not universally dominate; problem structure, parameter tuning, and restarts strongly affect which method achieves better global solutions.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

B. N. Alajmi, Nabil A. Ahmed, I. Abdelsalam and M. Marei. "A Robust MPPT Algorithm for PV Systems Using Advanced Hill Climbing and Simulated Annealing Techniques." *Electronics* (2025). [https://doi.org/10.3390/electronics14183644](https://doi.org/10.3390/electronics14183644)

S. Jacobson and E. Yücesan. "Analyzing the Performance of Generalized Hill Climbing Algorithms." *Journal of Heuristics*, 10 (2004): 387-405. [https://doi.org/10.1023/b:heur.0000034712.48917.a9](https://doi.org/10.1023/b:heur.0000034712.48917.a9)

Darrall Henderson, S. Jacobson and Alan W. Johnson. "The Theory and Practice of Simulated Annealing." \*\* (2006): 287-319. [https://doi.org/10.1007/0-306-48056-5\_10](https://doi.org/10.1007/0-306-48056-5_10)

M. Alrefaei and S. Andradóttir. "A Simulated Annealing Algorithm with Constant Temperature for Discrete Stochastic Optimization." *Management Science*, 45 (1999): 748-764. [https://doi.org/10.1287/mnsc.45.5.748](https://doi.org/10.1287/mnsc.45.5.748)

A. Dekkers and E. Aarts. "Global optimization and simulated annealing." *Mathematical Programming*, 50 (1991): 367-393. [https://doi.org/10.1007/bf01594945](https://doi.org/10.1007/bf01594945)

Alexander G. Nikolaev and S. Jacobson. "Simulated Annealing." \*\* (2011). [https://doi.org/10.1007/978-1-4419-1665-5\_1](https://doi.org/10.1007/978-1-4419-1665-5_1)

William L. Goffe, Gary D. Ferrier and J. E. Rogers. "Global Optimization of Statistical Functions with Simulated Annealing." *Journal of Econometrics*, 60 (1994): 65-99. [https://doi.org/10.1016/0304-4076(94)90038-8](https://doi.org/10.1016/0304-4076\(94\)90038-8)

J. E. Orosz and S. Jacobson. "Finite-Time Performance Analysis of Static Simulated Annealing Algorithms." *Computational Optimization and Applications*, 21 (2002): 21-53. [https://doi.org/10.1023/a:1013544329096](https://doi.org/10.1023/a:1013544329096)

Alan W. Johnson and S. Jacobson. "On the convergence of generalized hill climbing algorithms." *Discret. Appl. Math.*, 119 (2002): 37-57. [https://doi.org/10.1016/s0166-218x(01)00264-5](https://doi.org/10.1016/s0166-218x\(01\)00264-5)

Alan W. Johnson and S. Jacobson. "A class of convergent generalized hill climbing algorithms." *Appl. Math. Comput.*, 125 (2002): 359-373. [https://doi.org/10.1016/s0096-3003(00)00137-5](https://doi.org/10.1016/s0096-3003\(00\)00137-5)

Guoliang Liu, Shijie Han, Xiu-hai Zhao, J. Nelson, Hongshu Wang and Weiying Wang. "Optimisation algorithms for spatially constrained forest planning." *Ecological Modelling*, 194 (2006): 421-428. [https://doi.org/10.1016/j.ecolmodel.2005.10.028](https://doi.org/10.1016/j.ecolmodel.2005.10.028)

J. E. Orosz and S. Jacobson. "Analysis of Static Simulated Annealing Algorithms." *Journal of Optimization Theory and Applications*, 115 (2002): 165-182. [https://doi.org/10.1023/a:1019633214895](https://doi.org/10.1023/a:1019633214895)

\---

# **Yes, Bayesian optimization can substantially improve efficiency in combinatorial map design when evaluations are expensive.**

**Figure 1:** Consensus on BO improving combinatorial design efficiency

## **How Bayesian Optimization Improves Efficiency**

**1\. Fewer expensive evaluations**

* Combinatorial BO methods like COMBO, MerCBO, and random-mapping BO are explicitly designed for discrete, exponentially large spaces and consistently reach better solutions with *far fewer* evaluations than alternative methods (Deshwal et al. 2020, 7210-7218; Baptista and Poloczek 2018; Oh et al. 2019, 2910-2920; Kim et al. 2020).  
* In molecular and protein design, BO methods over massive sequence/graph spaces discover high‑quality designs much faster than heuristic or random search, demonstrating strong sample efficiency in domains structurally similar to combinatorial map layouts (Deshwal et al. 2020, 7210-7218; Baptista and Poloczek 2018; Bal et al. 2024).

**2\. Handling large, structured combinatorial spaces**

* COMBO builds a Gaussian process on a combinatorial graph and uses a diffusion kernel with variable selection, enabling scalable search over high‑dimensional categorical/ordinal spaces (Oh et al. 2019, 2910-2920).  
* MerCBO constructs explicit feature maps for diffusion kernels on discrete objects, combined with Thompson sampling, and matches or outperforms prior combinatorial BO methods on diverse benchmarks (Deshwal et al. 2020, 7210-7218).  
* Random-mapping approaches embed discrete configurations into a continuous polytope so that standard BO machinery can operate efficiently, with regret guarantees and empirical gains over existing methods (Kim et al. 2020).  
* Graph‑subset BO methods show that mapping subsets to a combinatorial graph and using local BO models can efficiently traverse very large design spaces (Liang et al. 2024).

**3\. Evidence from design and layout tasks**

* BO has improved design efficiency in process, material, structural, and circuit design by reducing required simulations or experiments compared with traditional or heuristic search, sometimes by large factors (e.g., 53% fewer iterations vs. pattern search in ball‑map–like chip design) (Paulson and Tsay 2024; Coelho et al. 2025; Zhang et al. 2021, 1-14; Shen et al. 2021, 1-3).

### **Takeaway for combinatorial map design**

For board‑game–style combinatorial maps (discrete tiles, adjacency constraints, expensive simulation/play‑test evaluation), these results strongly suggest BO can **achieve comparable or better map quality with many fewer evaluations** than naive or purely heuristic search.

## **Conclusion**

Bayesian optimization methods specialized for combinatorial spaces (COMBO, MerCBO, GameOpt, random‑mapping BO) provide clear efficiency gains in analogous high‑dimensional design problems. Applied to combinatorial map design, they are well‑justified choices when evaluation is costly and sample efficiency matters.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Aryan Deshwal, Syrine Belakaria and J. Doppa. "Mercer Features for Efficient Combinatorial Bayesian Optimization." \*\* (2020): 7210-7218. [https://doi.org/10.1609/aaai.v35i8.16886](https://doi.org/10.1609/aaai.v35i8.16886)

J. Paulson and Calvin Tsay. "Bayesian optimization as a flexible and efficient design framework for sustainable process systems." *ArXiv*, abs/2401.16373 (2024). [https://doi.org/10.48550/arxiv.2401.16373](https://doi.org/10.48550/arxiv.2401.16373)

R.P. Cardoso Coelho, A. F. Carvalho Alves, T.M. Nogueira Pires and F.M. Andrade Pires. "A composite Bayesian optimisation framework for material and structural design." *Computer Methods in Applied Mechanics and Engineering* (2025). [https://doi.org/10.1016/j.cma.2024.117516](https://doi.org/10.1016/j.cma.2024.117516)

Huidong Liang, Xingchen Wan and Xiaowen Dong. "Bayesian Optimization of Functions over Node Subsets in Graphs." *ArXiv*, abs/2405.15119 (2024). [https://doi.org/10.48550/arxiv.2405.15119](https://doi.org/10.48550/arxiv.2405.15119)

R. Baptista and Matthias Poloczek. "Bayesian Optimization of Combinatorial Structures." *ArXiv*, abs/1806.08838 (2018).

Changyong Oh, J. Tomczak, E. Gavves and M. Welling. "Combinatorial Bayesian Optimization using the Graph Cartesian Product." \*\* (2019): 2910-2920.

Shuhan Zhang, Fan Yang, Changhao Yan, Dian Zhou and Xuan Zeng. "An Efficient Batch-Constrained Bayesian Optimization Approach for Analog Circuit Synthesis via Multiobjective Acquisition Ensemble." *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 41 (2021): 1-14. [https://doi.org/10.1109/tcad.2021.3054811](https://doi.org/10.1109/tcad.2021.3054811)

Jungtaek Kim, Minsu Cho and Seungjin Choi. "Combinatorial Bayesian Optimization with Random Mapping Functions to Convex Polytope." *ArXiv*, abs/2011.13094 (2020).

Zan Shen, Xiaochun Li and Junfa Mao. "An Efficient Method for Ball Map Design Based on Bayesian Optimization." *2021 International Conference on Microwave and Millimeter Wave Technology (ICMMT)* (2021): 1-3. [https://doi.org/10.1109/icmmt52847.2021.9618132](https://doi.org/10.1109/icmmt52847.2021.9618132)

Melis Ilayda Bal, Pier Giuseppe Sessa, Mojmír Mutný and Andreas Krause. "Optimistic Games for Combinatorial Bayesian Optimization with Application to Protein Design." *ArXiv*, abs/2409.18582 (2024). [https://doi.org/10.48550/arxiv.2409.18582](https://doi.org/10.48550/arxiv.2409.18582)

\---

# **While distance-weighted gravity models predict healthcare accessibility reasonably well, they capture *potential* rather than fully realized access.**

**Figure 1:** Research support for gravity models in healthcare

## **How accurate are gravity-based healthcare accessibility models?**

* A 2023 systematic review (43 methodological, 309 empirical studies) finds gravity and 2SFCA-family models are the **dominant approach** for measuring potential spatial healthcare access; they are widely regarded as more realistic than simple ratios or nearest-facility measures because they combine supply, demand, and distance decay (Stacherl and Sauzet 2023).  
* Many applications use gravity-type measures descriptively (mapping underserved areas, comparing regions), where they provide nuanced and policy-relevant accessibility patterns (Stacherl and Sauzet 2023; Schuurman et al. 2010, 29-45; Yang et al. 2006, 23-32; Raju et al. 2020, 6-20).

## **Quantitative validation against observed use**

Where validated against actual utilization or flows, gravity-based models show **moderate predictive performance**:

* A modified gravity model calibrated with real road travel times and facility capacities in Russia achieved **R² ≈ 0.25 and MAPE \< 28%** for outpatient visits, considered interpretable and practically useful for planning (Begicheva and Begicheva 2025).  
* Gravity-based patient-flow models can approximate hospital choice and inter-municipal patient mobility with reasonably high rank correlation to observed visits, though individual-level behavior remains imperfectly captured (Begicheva 2025; Latruwe et al. 2022, 452-467).  
* Gravity-type accessibility indices correlate with broader development measures (e.g., local Human Development Index) and with health outcomes/utilization, supporting construct validity rather than point prediction accuracy (Zúñiga-Macías et al. 2025; Gao et al. 2021).

## **Key strengths and limits**

| Aspect | Evidence-based insight | Citations |
| ----- | ----- | ----- |
| What they best measure | **Potential** spatial access (who could reach what, given distance & capacity) | (Stacherl and Sauzet 2023; Schuurman et al. 2010, 29-45; Yang et al. 2006, 23-32; Luo and Qi 2009, |

     1100-7

)| | Factors improving accuracy | Network travel times, calibrated distance decay, capacity & competition terms | (Begicheva and Begicheva 2025; Sun et al. 2024; Luo and Qi 2009, 1100-7 ; Zhou et al. 2020, 394; Irlacher et al. 2023)| | Main limitations | Do not fully capture preferences, quality, socio-economic barriers; accuracy is moderate, not exact | (Stacherl and Sauzet 2023; Begicheva 2025; Latruwe et al. 2022, 452-467; Gao et al. 2021)|

**Figure 2:** Factors shaping accuracy of gravity accessibility models

## **Conclusion**

Distance-weighted gravity models are well-established, empirically supported tools that **approximate potential healthcare accessibility with moderate but useful accuracy**, especially for regional planning and equity analysis. They should not be treated as exact predictors of individual service use, but as robust proxies that work best when carefully calibrated and combined with contextual data.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

B. Stacherl and Odile Sauzet. "Gravity models for potential spatial healthcare access measurement: a systematic methodological review." *International Journal of Health Geographics*, 22 (2023). [https://doi.org/10.1186/s12942-023-00358-z](https://doi.org/10.1186/s12942-023-00358-z)

S. Begicheva and A. Begicheva. "Modified gravity model for assessing healthcare accessibility: problem statement, algorithm, and implementation." *Journal Of Applied Informatics* (2025). [https://doi.org/10.37791/2687-0649-2025-20-5-4-21](https://doi.org/10.37791/2687-0649-2025-20-5-4-21)

Shijie Sun, Qun Sun, Fubing Zhang and Jingzhen Ma. "A Spatial Accessibility Study of Public Hospitals: A Multi-Mode Gravity-Based Two-Step Floating Catchment Area Method." *Applied Sciences* (2024). [https://doi.org/10.3390/app14177713](https://doi.org/10.3390/app14177713)

S. Begicheva. "Spatial Modeling of Healthcare Accessibility Using Probabilistic Methods: A Case Study of Sverdlovsk Oblast." *AlterEconomics* (2025). [https://doi.org/10.31063/altereconomics/2025.22-2.9](https://doi.org/10.31063/altereconomics/2025.22-2.9)

Timo Latruwe, Marlies Van der Wee, P. Vanleenhove, Kwinten Michielsen, S. Verbrugge and D. Colle. "Improving inpatient and daycare admission estimates with gravity models." *Health Services and Outcomes Research Methodology*, 23 (2022): 452-467. [https://doi.org/10.1007/s10742-022-00298-4](https://doi.org/10.1007/s10742-022-00298-4)

N. Schuurman, Myriam Bérubé and Valorie A. Crooks. "Measuring potential spatial access to primary health care physicians using a modified gravity model." *Canadian Geographer*, 54 (2010): 29-45. [https://doi.org/10.1111/j.1541-0064.2009.00301.x](https://doi.org/10.1111/j.1541-0064.2009.00301.x)

Román Zúñiga-Macías, Fernando I. Becerra-López, Abel Palafox González and Humberto Gutiérrez-Pulido. "A connectivity-based index of healthcare accessibility using a spatial interaction approach." *Discover Public Health*, 22 (2025). [https://doi.org/10.1186/s12982-025-01063-x](https://doi.org/10.1186/s12982-025-01063-x)

Duck-Hye Yang, Robert M. Goerge and R. Mullner. "Comparing GIS-Based Methods of Measuring Spatial Accessibility to Health Services." *Journal of Medical Systems*, 30 (2006): 23-32. [https://doi.org/10.1007/s10916-006-7400-5](https://doi.org/10.1007/s10916-006-7400-5)

S. Raju, Shayesta Wajid, N. Radhakrishnan and S. Mathew. "Accessibility Analysis for Healthcare Centers using Gravity Model and Geospatial Techniques." *Tema. Journal of Land Use, Mobility and Environment*, 13 (2020): 6-20. [https://doi.org/10.6092/1970-9870/6414](https://doi.org/10.6092/1970-9870/6414)

W. Luo and Y. Qi. "An enhanced two-step floating catchment area (E2SFCA) method for measuring spatial accessibility to primary care physicians.." *Health & place*, 15 4 (2009): 1100-7. [https://doi.org/10.1016/j.healthplace.2009.06.002](https://doi.org/10.1016/j.healthplace.2009.06.002)

Xinxin Zhou, Zhaoyuan Yu, Linwang Yuan, Lei Wang and Changbin Wu. "Measuring Accessibility of Healthcare Facilities for Populations with Multiple Transportation Modes Considering Residential Transportation Mode Choice." *ISPRS Int. J. Geo Inf.*, 9 (2020): 394\. [https://doi.org/10.3390/ijgi9060394](https://doi.org/10.3390/ijgi9060394)

Michael Irlacher, Dieter Pennerstorfer, A. Renner and Florian Unger. "MODELING INTER‐REGIONAL PATIENT MOBILITY: THEORY AND EVIDENCE FROM SPATIALLY EXPLICIT DATA." *International Economic Review* (2023). [https://doi.org/10.1111/iere.12635](https://doi.org/10.1111/iere.12635)

F. Gao, Clara languille, Khalil karzazi, Mélanie Guhl, Baptiste Boukebous and S. Deguen. "Efficiency of fine scale and spatial regression in modelling associations between healthcare service spatial accessibility and their utilization." *International Journal of Health Geographics*, 20 (2021). [https://doi.org/10.1186/s12942-021-00276-y](https://doi.org/10.1186/s12942-021-00276-y)

\---

# **Combining spatial autocorrelation (Moran’s I) with distributive fairness indices enhances the detection of structural inequality by revealing both the magnitude and spatial patterning of disparities.**

## **1\. Introduction**

Structural inequality is often measured using distributive fairness indices such as the Gini coefficient, Theil index, or Jain’s Fairness Index (JFI), which quantify overall disparity but ignore spatial patterns. Spatial autocorrelation metrics like Moran’s I capture clustering or dispersion of values across space, but do not measure the magnitude of inequality itself. Recent research demonstrates that integrating these two approaches—using both spatial autocorrelation and distributive fairness indices—provides a more nuanced and powerful framework for detecting and understanding structural inequalities. This combination allows researchers to identify not only how unequal a distribution is, but also whether disadvantage is geographically clustered, which is crucial for policy targeting and understanding systemic inequities (Rey and Smith 2013, 55-70; Panzera and Postiglione 2019, 379-394; Zhu et al. 2018; Sui et al. 2019; Sangkasem and Puttanapong 2021; Yin et al. 2018, 50-62). Empirical studies in health, education, economic development, and environmental justice consistently show that joint application of these metrics uncovers hidden clusters of deprivation or privilege that scalar indices alone would miss (Dong et al. 2023; Flynn et al. 2021; Zhu et al. 2018; Verbeek 2018; Sui et al. 2019; Sangkasem and Puttanapong 2021). Some recent methodological advances even propose hybrid or decomposed indices that explicitly partition inequality into spatial and non-spatial components (Rey and Smith 2013, 55-70; Panzera and Postiglione 2019, 379-394). However, direct integration with JFI specifically remains rare; most work uses Gini or Theil as the distributive metric. Overall, the literature supports the value of combining spatial autocorrelation with fairness/inequality indices for improved detection and diagnosis of structural inequality.

**Figure 1:** Consensus on combining spatial and inequality measures.

## **2\. Methods**

A comprehensive search was conducted over 170 million research papers in Consensus, including sources from Semantic Scholar, PubMed, and others. A total of 988 potentially relevant papers were identified through targeted queries on combinations of spatial autocorrelation (Moran’s I) and distributive fairness/inequality indices (including JFI). After de-duplication and screening for relevance to structural inequality detection, 434 papers were deemed eligible. The top 50 most relevant papers were included in this review.

| Identification | Screening | Eligibility | Included |
| ----- | ----- | ----- | ----- |
| 988 | 641 | 434 | 50 |

**Figure 2:** Flowchart showing paper selection process for this review. Eight unique search strategies were used to capture literature on both foundational concepts and applied integrations of spatial autocorrelation with fairness/inequality metrics.

## **3\. Results**

### **3.1 Limitations of Scalar Inequality Indices Alone**

Traditional distributive fairness indices (Gini, Theil, JFI) are “aspatial”—they summarize overall disparity but ignore where advantage/disadvantage occurs (Panzera and Postiglione 2019, 379-394; Zhu et al. 2018; Netrdová and Nosek 2009, 52-65; Sui et al. 2019). This anonymity property means very different spatial patterns can yield identical index values (Panzera and Postiglione 2019, 379-394). As a result, scalar indices may miss critical information about clustering or segregation that underpins structural inequality (Rey and Smith 2013, 55-70; Panzera and Postiglione 2019, 379-394).

### **3.2 Added Value from Spatial Autocorrelation Metrics**

Spatial autocorrelation statistics like Moran’s I reveal whether high or low values are clustered or dispersed across space (Bivand and Wong 2018, 716 \- 748; Chen 2020; Anselin 2010, 93-115). When applied alongside fairness indices, they can identify “hot spots” or “cold spots” of deprivation/privilege even when overall inequality appears moderate (Dong et al. 2023; Flynn et al. 2021; Zhu et al. 2018; Verbeek 2018). For example, studies in healthcare resource allocation found that while Gini/Theil indicated declining overall inequality over time, Moran’s I revealed persistent clusters of under-resourced areas requiring targeted intervention (Dong et al. 2023; Flynn et al. 2021).

### **3.3 Integrated Approaches: Decomposition & Hybrid Indices**

Recent methodological advances include:

* **Spatial decomposition of Gini**: Partitions total inequality into between-neighbor (spatial) and non-neighbor (aspatial) components using a spatial weights matrix (Rey and Smith 2013, 55-70).  
* **Gini-correlation-based measures**: Jointly account for magnitude and spatial patterning; allow identification of regional contributions to overall inequality (Panzera and Postiglione 2019, 379-394).  
* **Spatial regression models**: Incorporate both scalar disparity measures and spatial dependence to better explain observed patterns in health/environmental justice outcomes (Verbeek 2018; Martines et al. 2020, 7 \- 36). These approaches consistently outperform single-metric analyses in detecting structurally patterned inequities.

### **3.4 Applications Across Domains**

* **Healthcare**: Joint use reveals clusters of poor access despite improving national averages (Dong et al. 2023; Flynn et al. 2021; Zhu et al. 2018).  
* **Education**: Spatial clustering analysis identifies persistent low-attainment regions masked by average equity scores (Sandu et al. 2025).  
* **Environmental justice**: Spatial regression models uncover income-related exposure clusters invisible to standard regressions (Verbeek 2018).  
* **Urban development**: Combined metrics highlight core-periphery structures not captured by scalar indices alone (Roy et al. 2024, 81 \- 104; Dey et al. 2024).

#### **Results Timeline**

* **2008**  
  * 1 paper: (Netrdová and Nosek 2009, 52-65)- **2013**  
  * 2 papers: (Rey and Smith 2013, 55-70; Chen 2013)- **2018**  
  * 3 papers: (Bivand and Wong 2018, 716 \- 748; Zhu et al. 2018; Verbeek 2018)- **2019**  
  * 1 paper: (Panzera and Postiglione 2019, 379-394)- **2020**  
  * 2 papers: (Chen 2020; Iyer et al. 2020)- **2021**  
  * 2 papers: (Flynn et al. 2021; Majumdar et al. 2021, 6-18)- **2022**  
  * 3 papers: (Chen 2022; Libório et al. 2022, 185-211; Lu et al. 2022)- **2023**  
  * 4 papers: (Qu et al. 2023; Dong et al. 2023; Purwaningsih and Suseno 2023; Zhang et al. 2023, 3355 \- 3368)- **2024**  
  * 1 paper: (Roy et al. 2024, 81 \- 104)- **2025**  
  * 1 paper: (Owasitth et al. 2025\)**Figure 3:** Timeline showing evolution of integrated spatial–inequality methods since early 2010s. Larger markers indicate more citations.

#### **Top Contributors**

| Type | Name | Papers |
| ----- | ----- | ----- |
| Author | Yanguang Chen | (Chen 2022; Chen 2020; Chen 2013; Chen 2025\) |
| Author | S. Rey | (Rey and Smith 2013, 55-70; Rey and Smith 2012, 55 \- 70; Rey 2001\) |
| Author | Enhong Dong | (Dong et al. 2023\) |
| Journal | *PLoS ONE* | (Chen 2020; Chen 2013\) |
| Journal | *Letters in Spatial and Resource Sciences* | (Rey and Smith 2013, 55-70; Rey and Smith 2012, 55 \- 70\) |
| Journal | *International Journal of Environmental Research and Public Health* | (Zhu et al. 2018; Lu et al. 2022\) |

**Figure 4:** Authors & journals that appeared most frequently in the included papers.

## **4\. Discussion**

The reviewed literature strongly supports the complementary use of global spatial autocorrelation (Moran’s I) with distributive fairness/inequality indices for detecting structural inequalities across domains such as health care access, education outcomes, economic development, and environmental justice (Dong et al. 2023; Rey and Smith 2013, 55-70; Flynn et al. 2021; Panzera and Postiglione 2019, 379-394; Zhu et al. 2018; Verbeek 2018). Scalar indices alone risk missing critical information about geographic clustering—a hallmark of systemic inequity—while Moran’s I alone cannot quantify the magnitude or severity of disparities.

Integrated approaches—such as spatial decomposition frameworks or hybrid correlation-based measures—provide richer diagnostics by partitioning observed disparities into their spatially structured versus random components (Rey and Smith 2013, 55-70; Panzera and Postiglione 2019, 379-394). These methods have been shown empirically to reveal persistent clusters (“hot spots”) even when overall disparity appears to decline over time—a key insight for policy targeting.

However, direct integration with Jain’s Fairness Index (JFI) specifically remains rare; most studies use Gini or Theil as their primary distributive metric. There is also ongoing debate about optimal weighting schemes for composite indicators when incorporating spatial dependence (Libório et al. 2022, 185-211), as well as challenges related to data availability at fine geographic scales.

### **Claims & Evidence Table**

| Claim | Evidence Strength | Reasoning | Papers |
| ----- | ----- | ----- | ----- |
| Combining Moran's I with fairness/inequality indices improves detection of structural inequality | Evidence strength: Strong (9/10) | Multiple empirical studies show joint use reveals clustered disadvantage missed by scalar metrics alone | (Dong et al. 2023; Rey and Smith 2013, 55-70; Flynn et al. 2021; Panzera and Postiglione 2019, 379-394; Zhu et al. 2018\) |
| Scalar fairness/inequality indices are insufficient for diagnosing geographic clustering | Evidence strength: Strong (8/10) | Aspatial measures can yield same value for very different patterns; fail to detect segregation/hotspots | (Panzera and Postiglione 2019, 379-394; Zhu et al. 2018; Netrdová and Nosek 2009, 52-65) |
| Integrated decomposition/hybrid approaches provide richer diagnostics than single-metric analyses | Evidence strength: Strong (8/10) | Decomposition frameworks partition total disparity into spatial/non-spatial components; empirically validated | (Rey and Smith 2013, 55-70; Panzera and Postiglione 2019, 379-394) |
| Direct integration with JFI specifically is rare; Gini/Theil more common | Evidence strength: Moderate (5/10) | Most studies reviewed use Gini/Theil; few mention JFI directly | (Qu et al. 2023; Libório et al. 2022, 185-211) |
| Optimal weighting schemes for composite indicators remain debated | Evidence strength: Moderate (5/10) | Different weighting affects sensitivity to spatial dependence; no consensus yet on best practice | (Libório et al. 2022, 185-211) |
| Single-metric approaches may mislead policy if used alone | Evidence strength: Moderate (6/10) | Policy based only on scalar index may overlook persistent clusters needing intervention | (Dong et al. 2023; Flynn et al. 2021\) |

**Figure undefined:** Key claims and support evidence identified in these papers.

## **5\. Conclusion**

Combining global spatial autocorrelation (Moran's I) with distributive fairness/inequality indices significantly enhances the detection and diagnosis of structural inequalities by capturing both magnitude and geographic patterning. While direct integration with JFI is still emerging in the literature compared to Gini/Theil-based approaches, the conceptual framework is robustly supported across multiple domains.

### **Research Gaps**

Despite strong evidence supporting integrated approaches using Moran's I plus Gini/Theil-type metrics across health care, education, environment, and economics domains—and some methodological advances—there are gaps regarding:

* Direct application/integration with Jain's Fairness Index (JFI)  
* Standardized best practices for composite indicator weighting  
* Fine-scale data availability for certain regions/outcomes  
* Application beyond traditional domains (e.g., algorithmic bias/fairness)

#### **Research Gaps Matrix**

| Topic/Outcome | Health Care Access | Education Outcomes | Economic Development | Environmental Justice |
| ----- | ----- | ----- | ----- | ----- |
| Scalar index only | **10** | **5** | **8** | **6** |
| Moran's I only | **8** | **2** | **5** | **5** |
| Combined approach | **9** | **2** | **5** | **2** |
| Direct JFI integration | **GAP** | **GAP** | **GAP** | **GAP** |

**Figure undefined:** Matrix showing prevalence of different analytic approaches by domain; direct JFI integration remains a gap.

### **Open Research Questions**

Future research should focus on developing standardized frameworks that integrate JFI directly with spatial autocorrelation metrics like Moran's I across diverse domains.

| Question | Why |
| ----- | ----- |
| **How can Jain's Fairness Index be directly integrated with Moran's I for structural inequality analysis?** | Direct integration could enable broader adoption in fields where JFI is preferred over Gini/Theil. |
| **What are optimal weighting schemes when constructing composite indicators sensitive to both magnitude and location?** | Weighting affects sensitivity to clustering vs overall disparity; best practices remain unsettled. |
| **How can integrated metrics be adapted for algorithmic bias/fairness detection in machine learning applications?** | Extending these tools beyond traditional domains could improve equity assessment in automated systems. |

**Figure undefined:** Open questions highlight directions for methodological innovation.

In summary: Integrating global spatial autocorrelation with distributive fairness/inequality metrics provides a more complete picture of structural inequalities than either approach alone—especially important for effective policy targeting—but further work is needed on direct JFI integration and methodological standardization.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

R. Qu, Sang-Hyun Lee, Zaewoong Rhee and Seung-Jong Bae. "Analysis of Inequality Levels of Industrial Development in Rural Areas through Inequality Indices and Spatial Autocorrelation." *Sustainability* (2023). [https://doi.org/10.3390/su15108102](https://doi.org/10.3390/su15108102)

Yanguang Chen. "Spatial autocorrelation equation based on Moran’s index." *Scientific Reports*, 13 (2022). [https://doi.org/10.1038/s41598-023-45947-x](https://doi.org/10.1038/s41598-023-45947-x)

Enhong Dong, Xiaoting Sun, Ting Xu, Shixiang Zhang, Tao Wang, Lufa Zhang and Weimin Gao. "Measuring the inequalities in healthcare resource in facility and workforce: A longitudinal study in China." *Frontiers in Public Health*, 11 (2023). [https://doi.org/10.3389/fpubh.2023.1074417](https://doi.org/10.3389/fpubh.2023.1074417)

S. Rey and Richard J. Smith. "A spatial decomposition of the Gini coefficient." *Letters in Spatial and Resource Sciences*, 6 (2013): 55-70. [https://doi.org/10.1007/s12076-012-0086-z](https://doi.org/10.1007/s12076-012-0086-z)

R. Bivand and David W. S. Wong. "Comparing implementations of global and local indicators of spatial association." *TEST*, 27 (2018): 716 \- 748\. [https://doi.org/10.1007/s11749-018-0599-x](https://doi.org/10.1007/s11749-018-0599-x)

Cheryl J. Flynn, S. Majumdar and Ritwik Mitra. "Evaluating Fairness in the Presence of Spatial Autocorrelation." *ArXiv*, abs/2101.01703 (2021).

Domenica Panzera and Paolo Postiglione. "Measuring the Spatial Dimension of Regional Inequality: An Approach Based on the Gini Correlation Measure." *Social Indicators Research*, 148 (2019): 379-394. [https://doi.org/10.1007/s11205-019-02208-7](https://doi.org/10.1007/s11205-019-02208-7)

Yanguang Chen. "An analytical process of spatial autocorrelation functions based on Moran’s index." *PLoS ONE*, 16 (2020). [https://doi.org/10.1371/journal.pone.0249589](https://doi.org/10.1371/journal.pone.0249589)

Yanguang Chen. "New Approaches for Calculating Moran’s Index of Spatial Autocorrelation." *PLoS ONE*, 8 (2013). [https://doi.org/10.1371/journal.pone.0068336](https://doi.org/10.1371/journal.pone.0068336)

H. Iyer, J. Flanigan, N. Wolf, L. Schroeder, S. Horton, M. Castro and T. Rebbeck. "Geospatial evaluation of trade-offs between equity in physical access to healthcare and health systems efficiency." *BMJ Global Health*, 5 (2020). [https://doi.org/10.1136/bmjgh-2020-003493](https://doi.org/10.1136/bmjgh-2020-003493)

Subham Roy, Suranjan Majumder, Arghadeep Bose and Indrajit Roy Chowdhury. "Spatial heterogeneity in the urban household living conditions: A-GIS-based spatial analysis." *Annals of GIS*, 30 (2024): 81 \- 104\. [https://doi.org/10.1080/19475683.2024.2304194](https://doi.org/10.1080/19475683.2024.2304194)

Silvia Purwaningsih and D. Suseno. "Analysis of Spatial Autocorrelation and Causality GRDP and Income Inequality in Java." *Efficient: Indonesian Journal of Development Economics* (2023). [https://doi.org/10.15294/efficient.v6i1.57599](https://doi.org/10.15294/efficient.v6i1.57599)

Bin Zhu, C. Hsieh and Yue Zhang. "Incorporating Spatial Statistics into Examining Equity in Health Workforce Distribution: An Empirical Analysis in the Chinese Context." *International Journal of Environmental Research and Public Health*, 15 (2018). [https://doi.org/10.3390/ijerph15071309](https://doi.org/10.3390/ijerph15071309)

M. Libório, J. F. de Abreu, P. Ekel and A. Machado. "Effect of sub-indicator weighting schemes on the spatial dependence of multidimensional phenomena." *Journal of Geographical Systems*, 25 (2022): 185-211. [https://doi.org/10.1007/s10109-022-00401-w](https://doi.org/10.1007/s10109-022-00401-w)

P. Netrdová and V. Nosek. "Approaches to measuring the relevance of geographical dimension of societa inequalities." *Geografie*, 114 (2009): 52-65. [https://doi.org/10.37040/geografie2009114010052](https://doi.org/10.37040/geografie2009114010052)

S. Majumdar, Cheryl J. Flynn and Ritwik Mitra. "Detecting Bias in the Presence of Spatial Autocorrelation." \*\* (2021): 6-18.

Rattapong Owasitth, S. Lawpoolsri, Risa Chaisuparat, Poolpruek Soparat and Palinee Detsomboonrat. "Applications of spatial autocorrelation in global dentistry: a scoping review." *BMC Oral Health*, 25 (2025). [https://doi.org/10.1186/s12903-025-07034-7](https://doi.org/10.1186/s12903-025-07034-7)

Wei Lu, Yuechen Li, Rongkun Zhao, Bo He and Zihua Qian. "Spatial Pattern and Fairness Measurement of Educational Resources in Primary and Middle Schools: A Case Study of Chengdu–Chongqing Economic Circle." *International Journal of Environmental Research and Public Health*, 19 (2022). [https://doi.org/10.3390/ijerph191710840](https://doi.org/10.3390/ijerph191710840)

T. Verbeek. "Unequal residential exposure to air pollution and noise: A geospatial environmental justice analysis for Ghent, Belgium." *SSM \- Population Health*, 7 (2018). [https://doi.org/10.1016/j.ssmph.2018.100340](https://doi.org/10.1016/j.ssmph.2018.100340)

Ce Zhang, Wangyong Lv, Ping Zhang and Jiacheng Song. "Multidimensional spatial autocorrelation analysis and it’s application based on improved Moran’s I." *Earth Science Informatics*, 16 (2023): 3355 \- 3368\. [https://doi.org/10.1007/s12145-023-01090-9](https://doi.org/10.1007/s12145-023-01090-9)

S. Rey and Richard J. Smith. "A spatial decomposition of the Gini coefficient." *Letters in Spatial and Resource Sciences*, 6 (2012): 55 \- 70\. [https://doi.org/10.1007/s12076-012-0086-z](https://doi.org/10.1007/s12076-012-0086-z)

Xinmin Sui, Yifan Yu and Liu Huhui. "Measurement of spatial equity : a case study of nursing institution." *Proceedings of the 55th ISOCARP World Planning Congress* (2019). [https://doi.org/10.47472/bgdi1793](https://doi.org/10.47472/bgdi1793)

M. Martines, Ricardo Vicente Ferreira, R. H. Toppa, L. M. Assunção, M. Desjardins and E. Delmelle. "Detecting space–time clusters of COVID-19 in Brazil: mortality, inequality, socioeconomic vulnerability, and the relative risk of the disease in Brazilian municipalities." *Journal of Geographical Systems*, 23 (2020): 7 \- 36\. [https://doi.org/10.1007/s10109-020-00344-0](https://doi.org/10.1007/s10109-020-00344-0)

Alexandra Sandu, Jennifer Keating, Katy Huxley, Tony Whiffen and Rob French. "Leveraging Administrative Data to Identify Spatial Inequalities in Educational Attainment across Wales." *International Journal of Population Data Science* (2025). [https://doi.org/10.23889/ijpds.v10i4.3286](https://doi.org/10.23889/ijpds.v10i4.3286)

Souvik Dey, J. Ray and Rajarshi Majumder. "Spatial Inequality in Sub-National Human Development Index: A Case Study of West Bengal Districts." *Sustainable Futures* (2024). [https://doi.org/10.1016/j.sftr.2024.100330](https://doi.org/10.1016/j.sftr.2024.100330)

Yanguang Chen. "Structural Decomposition of Moran's Index by Getis-Ord's Indices." \*\* (2025).

Krittaya Sangkasem and Nattapong Puttanapong. "Analysis of spatial inequality using DMSP‐OLS nighttime‐light satellite imageries: A case study of Thailand." *Regional Science Policy and Practice* (2021). [https://doi.org/10.1111/rsp3.12386](https://doi.org/10.1111/rsp3.12386)

C. Yin, Qingsong He, Yanfang Liu, Weiqiang Chen and Yuan Gao. "Inequality of public health and its role in spatial accessibility to medical facilities in China." *Applied Geography*, 92 (2018): 50-62. [https://doi.org/10.1016/j.apgeog.2018.01.011](https://doi.org/10.1016/j.apgeog.2018.01.011)

L. Anselin. "Local Indicators of Spatial Association—LISA." *Geographical Analysis*, 27 (2010): 93-115. [https://doi.org/10.1111/j.1538-4632.1995.tb00338.x](https://doi.org/10.1111/j.1538-4632.1995.tb00338.x)

S. Rey. "Spatial Analysis of Regional Income Inequality." \*\* (2001).

\---

**Hexagonal grids can mitigate some edge-related artifacts, but they do not eliminate edge-effect bias in spatial autocorrelation.**

**Figure 1:** Consensus visualization for hexagon grids and edge effects.

## **How grid shape influences edge effects**

* Regular tessellations (triangles, squares, hexagons) induce **different distributions of Moran’s I and local Moran’s Ii**, even on the same underlying process (Boots and Tiefelsdorf 2000, 319-348). This includes effects related to boundaries, because edge cells have fewer neighbors and different connectivity.  
* Hexagons provide **more isotropic neighbor structure** (six equal neighbors), which reduces artificial corner vs edge asymmetries that are pronounced in square rook/queen schemes (Boots and Tiefelsdorf 2000, 319-348; Luo et al. 2018, 476). This tends to make local statistics near boundaries more geometrically consistent, but not unbiased.  
* In a simultaneous autoregressive model, variance functions for the spatial parameter ρ differ for square and hexagonal lattices; edge configuration and adjacency structure must be explicitly accounted for regardless of grid type, indicating that edge effects remain an issue under all tessellations (Luo et al. 2018, 476).

## **Evidence from related spatial tasks**

* For tasks such as identifying urban employment subcenters using Local Moran’s I on square, hexagonal, and triangular grids, **edge connections (corners vs edges) significantly affect detected clusters and compactness** (Monzur et al. 2024). Hexagons change how these edge artifacts appear but do not remove them.  
* In kriging design comparisons, hexagonal sampling can be more efficient than squares under some covariance conditions, but performance is still sensitive to sample layout and boundary configuration, again indicating that geometry helps but does not cure boundary-induced bias (Yfantis et al. 1987, 183-205).

## **Conclusion**

Hexagonal grids offer more uniform neighbors and can **reduce some geometric artifacts of square corners**, but edge-effect bias in spatial autocorrelation statistics still exists and must be addressed with explicit edge corrections or modeling choices, regardless of grid shape.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

B. Boots and M. Tiefelsdorf. "Global and local spatial autocorrelation in bounded regular tessellations." *Journal of Geographical Systems*, 2 (2000): 319-348. [https://doi.org/10.1007/pl00011461](https://doi.org/10.1007/pl00011461)

Tawhid Monzur, Tanzila Tabassum and Nawshin Bashir. "Alternative Tessellations for the Identification of Urban Employment Subcenters: A Comparison of Triangles, Squares, and Hexagons." *Journal of Geovisualization and Spatial Analysis*, 8 (2024). [https://doi.org/10.1007/s41651-024-00200-5](https://doi.org/10.1007/s41651-024-00200-5)

E. Yfantis, G. Flatman and J. Behar. "Efficiency of kriging estimation for square, triangular, and hexagonal grids." *Mathematical Geology*, 19 (1987): 183-205. [https://doi.org/10.1007/bf00897746](https://doi.org/10.1007/bf00897746)

Qing Luo, D. Griffith and Huayi Wu. "On the Statistical Distribution of the Nonzero Spatial Autocorrelation Parameter in a Simultaneous Autoregressive Model." *ISPRS Int. J. Geo Inf.*, 7 (2018): 476\. [https://doi.org/10.3390/ijgi7120476](https://doi.org/10.3390/ijgi7120476)

\---

