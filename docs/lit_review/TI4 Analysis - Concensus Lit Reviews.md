# **Evidence suggests Jain’s Fairness Index is limited for fine‑grained spatial equity in grids**

## **1\. Introduction**

Jain’s Fairness Index (JFI) is a widely used scalar metric for quantifying the evenness of resource allocation, especially in networked and grid-based systems. Its popularity stems from its axiomatic foundation, simplicity, and ability to summarize fairness in a single value (Lan et al. 2009, 1-9; Lan et al. 2010; Lan et al. 2009). However, when applied to discrete spatial grids—such as power distribution networks, microgrids, or spatially explicit resource allocation problems—questions arise about whether JFI can fully capture the nuances of spatial equity. Recent research highlights that while JFI effectively measures overall distributional fairness, it does not account for spatial configuration or multi-dimensional aspects of equity, which are often critical in grid-based applications (Liu et al. 2020, 4502-4512; Alyami 2024; Itote et al. 2025; Köppen et al. 2013, 841-846; He et al. 2024, 4750-4765; Dynge and Cali 2025; Gupta et al. 2024, 1-6). Alternative or complementary metrics (e.g., Gini index, entropy-based indices, spatial/social equity indices) are frequently employed to address these limitations (Itote et al. 2025; Köppen et al. 2013, 841-846; He et al. 2024, 4750-4765; Wismadi et al. 2009). This review synthesizes the literature on the accuracy and appropriateness of Jain’s Fairness Index for representing resource equity in discrete spatial grids.

**Figure 1:** Consensus on Jain index as fairness metric

## **2\. Methods**

A comprehensive search was conducted across over 170 million research papers indexed by Consensus, including Semantic Scholar and PubMed. The search strategy targeted foundational works on Jain’s Fairness Index, critiques and limitations in spatial contexts, alternative equity metrics (Gini/Theil), and interdisciplinary applications in energy, urban planning, and networked systems. In total, 997 papers were identified; after screening and eligibility filtering, 50 highly relevant papers were included in this review.

| Identification | Screening | Eligibility | Included |
| ----- | ----- | ----- | ----- |
| 997 | 635 | 510 | 50 |

**Figure 2:** Flow diagram of paper selection process for this review. Eight unique search groups were used to ensure broad coverage of fairness metrics and their application to discrete spatial grids.

## **3\. Results**

### **3.1 Axiomatic Foundations and Strengths of Jain’s Index**

JFI is grounded in a set of fairness axioms—continuity, symmetry, population-independence—and is Schur-concave, meaning it increases with more equal allocations (Lan et al. 2009, 1-9; Lan et al. 2010; Lan et al. 2009). It is widely adopted due to its scale-independence and ease of interpretation (Lan et al. 2009, 1-9; Singh et al. 2025, 61299-61311). In grid-based energy systems and communication networks, JFI is often embedded into optimization frameworks to balance efficiency with fairness (Alyami 2024; Zhou et al. 2025; Hossfeld et al. 2016, 184-187; Dynge and Cali 2025).

### **3.2 Application to Discrete Spatial Grids**

In practical grid scenarios (e.g., power distribution networks), JFI has been used to evaluate fairness among households or nodes regarding curtailment or resource allocation (Liu et al. 2020, 4502-4512; Alyami 2024; Itote et al. 2025; Gupta and Molzahn 2024). Studies show that while JFI can indicate overall evenness across nodes or locations, it does not distinguish between different spatial patterns that yield the same allocation vector but have different implications for access or vulnerability (Liu et al. 2020, 4502-4512; Itote et al. 2025).

### **3.3 Limitations: Spatial Configuration and Multi-Dimensional Equity**

Several studies highlight that JFI is agnostic to geometry—it evaluates only the allocation vector without considering adjacency or clustering effects (Itote et al. 2025; Köppen et al. 2013, 841-846). For example, two grids with identical per-cell resources but different patterns (clustered deprivation vs. dispersed) will have identical JFI scores but very different real-world implications (Itote et al. 2025). Multi-dimensional extensions (e.g., multi-Jain indices) have been proposed to capture proportionality, envy-freeness, or other fairness concepts beyond what a single scalar can provide (Köppen et al. 2013, 841-846).

### **3.4 Alternative Metrics for Spatial Equity**

Spatially explicit applications often supplement or replace JFI with other indices such as the Gini coefficient (for inequality), entropy-based measures (for diversity), accessibility indices (for reachability), or custom spatial/social equity indices that incorporate geographic features directly into the analysis (Köppen et al. 2013, 841-846; He et al. 2024, 4750-4765; Wismadi et al. 2009). These alternatives are better suited for capturing both distributional evenness and spatial configuration.

#### **Results Timeline**

* **2009**  
  * 1 paper: (Lan et al. 2009, 1-9)- **2010**  
  * 1 paper: (Lan et al. 2010)- **2012**  
  * 1 paper: (Sediq et al. 2012, 577-583)- **2013**  
  * 2 papers: (Köppen et al. 2013, 841-846; Sediq et al. 2013, 3496-3509)- **2016**  
  * 1 paper: (Hossfeld et al. 2016, 184-187)- **2018**  
  * 1 paper: (Zarabie et al. 2018, 6029-6040)- **2020**  
  * 1 paper: (Liu et al. 2020, 4502-4512)- **2022**  
  * 1 paper: (Blanco and G'azquez 2022, 106287)- **2024**  
  * 3 papers: (Alyami 2024; Gupta and Molzahn 2024; He et al. 2024, 4750-4765)- **2025**  
  * 8 papers: (Itote et al. 2025; Singh et al. 2025, 61299-61311; Zhou et al. 2025; AbdulSamadElSkaff et al. 2025, 1-6; Dataesatu et al. 2025, 1-7; Ghazi et al. 2025, 118515-118535; Dynge and Cali 2025; González-Sendino et al. 2025, 109979\)**Figure 3:** Timeline of key publications on Jain's index and spatial equity. Larger markers indicate more citations.

#### **Top Contributors**

| Type | Name | Papers |
| ----- | ----- | ----- |
| Author | T. Lan | (Lan et al. 2009, 1-9; Lan et al. 2010; Lan et al. 2009\) |
| Author | A. B. Sediq | (Sediq et al. 2013, 3496-3509; Sediq et al. 2012, 577-583) |
| Author | Rahul K. Gupta | (Gupta and Molzahn 2024; Gupta et al. 2024, 1-6) |
| Journal | *IEEE Transactions on Smart Grid* | (Liu et al. 2020, 4502-4512; Zarabie et al. 2018, 6029-6040) |
| Journal | *IEEE Access* | (Singh et al. 2025, 61299-61311; Ghazi et al. 2025, 118515-118535; Mehmood and Arshad 2020, 210733-210749) |
| Journal | *Applied Energy* | (Dynge and Cali 2025; Fraunholz et al. 2025\) |

**Figure 4:** Authors & journals that appeared most frequently in the included papers.

## **4\. Discussion**

The literature consistently finds that while Jain’s Fairness Index provides a robust measure of overall distributional fairness—especially when allocations are represented as vectors—it falls short when nuanced aspects of **spatial equity** are required (Lan et al. 2009, 1-9; Liu et al. 2020, 4502-4512; Itote et al. 2025). Its inability to account for geographic arrangement means it cannot distinguish between allocations that are numerically equal but geographically disparate—a critical limitation in many real-world grid applications where location matters for access or vulnerability (Itote et al. 2025; Köppen et al. 2013, 841-846).

Recent work proposes multi-metric approaches: using multiple per-entity features with separate JFI calculations or combining JFI with other indices like Gini or entropy-based measures to better capture proportionality and envy-freeness alongside simple evenness (Köppen et al. 2013, 841-846). In fields such as urban planning and public health—where spatial justice is paramount—researchers increasingly rely on Lorenz curves/Gini coefficients and custom accessibility metrics rather than solely on JFI (Köppen et al. 2013, 841-846; He et al. 2024, 4750-4765).

Overall research quality is high regarding theoretical foundations; however, empirical validation in complex real-world grids remains limited. There is consensus that no single scalar index—including JFI—can fully represent all dimensions of resource equity in discrete spatial settings.

### **Claims and Evidence Table**

| Claim | Evidence Strength | Reasoning | Papers |
| ----- | ----- | ----- | ----- |
| Jain’s Index robustly measures overall evenness/fairness | Evidence strength: Strong (9/10) | Strong axiomatic basis; widely validated across domains | (Lan et al. 2009, 1-9; Lan et al. 2010; Lan et al. 2009\) |
| Jain’s Index does not capture geographic/spatial configuration | Evidence strength: Strong (8/10) | Multiple studies show identical scores for different spatial patterns | (Liu et al. 2020, 4502-4512; Itote et al. 2025\) |
| Multi-dimensional/multi-metric approaches improve representation | Evidence strength: Moderate (6/10) | Extensions using multiple features/indices better reflect proportionality/envy-freeness | (Köppen et al. 2013, 841-846) |
| Alternative indices (Gini/entropy/accessibility) needed for full equity | Evidence strength: Moderate (6/10) | Spatial/social equity requires additional metrics beyond scalar evenness | (Köppen et al. 2013, 841-846; He et al. 2024, 4750-4765; Wismadi et al. 2009\) |
| Empirical studies show trade-offs between efficiency & fairness | Evidence strength: Moderate (5/10) | Optimization frameworks reveal trade-offs; no universal optimal solution | (Sediq et al. 2013, 3496-3509; Sediq et al. 2012, 577-583) |
| Single-index approaches risk missing critical aspects of justice | Evidence strength: Moderate (5/10) | Scalar indices may overlook clustering/vulnerability/access issues | (Itote et al. 2025; Wismadi et al. 2009\) |

**Figure undefined:** Key claims and support evidence identified in these papers.

## **5\. Conclusion**

Jain’s Fairness Index remains a valuable tool for summarizing how evenly resources are distributed across nodes in discrete grids but does not fully represent **spatial equity** due to its lack of sensitivity to geographic arrangement and multi-dimensional needs. For accurate assessment of resource equity in discrete spatial grids—especially where location-specific access matters—complementary use of additional metrics such as Gini coefficients or custom spatial/social indices is recommended.

### **Research Gaps**

| Topic/Outcome | Theoretical Analysis | Empirical Grid Studies | Multi-Metric Approaches | Spatial Pattern Sensitivity |
| ----- | ----- | ----- | ----- | ----- |
| Overall fairness/evenness | **10** | **6** | **5** | **2** |
| Geographic configuration | **2** | **1** | **3** | **8** |
| Multi-dimensional justice | **GAP** | **GAP** | **6** | **GAP** |

**Figure undefined:** Matrix showing research focus areas versus study attributes; gaps exist especially for empirical studies on multi-dimensional justice.

### **Open Research Questions**

Future work should focus on developing composite metrics that integrate both distributional evenness and explicit modeling of geographic/spatial relationships within discrete grids.

| Question | Why |
| ----- | ----- |
| **How can composite indices be developed to jointly capture both distributional evenness and explicit geographic configuration?** | Integrating these aspects would enable more accurate assessments of resource equity in real-world grids where both matter equally. |
| **What empirical impacts do different fairness metrics have on vulnerable populations within actual grid deployments?** | Understanding real-world consequences will guide metric selection toward outcomes aligned with social justice goals. |

**Figure undefined:** Open questions highlight directions for integrating multidimensional fairness into practical grid applications.

In summary: While Jain’s Fairness Index provides a strong baseline measure for overall resource equality in discrete grids, it should be complemented by additional metrics when true **spatial equity** is required.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

T. Lan, D. Kao, M. Chiang and A. Sabharwal. "An Axiomatic Theory of Fairness in Network Resource Allocation." *2010 Proceedings IEEE INFOCOM* (2009): 1-9. [https://doi.org/10.1109/infcom.2010.5461911](https://doi.org/10.1109/infcom.2010.5461911)

Michael Z. Liu, Andreas T. Procopiou, Kyriacos Petrou, L. Ochoa, Tom Langstaff, Justin Harding and J. Theunissen. "On the Fairness of PV Curtailment Schemes in Residential Distribution Networks." *IEEE Transactions on Smart Grid*, 11 (2020): 4502-4512. [https://doi.org/10.1109/tsg.2020.2983771](https://doi.org/10.1109/tsg.2020.2983771)

Saeed Alyami. "Fairness and usability analysis in renewable power curtailment: A microgrid network study using bankruptcy rules." *Renewable Energy* (2024). [https://doi.org/10.1016/j.renene.2024.120424](https://doi.org/10.1016/j.renene.2024.120424)

Francis Maina Itote, Ryuto Shigenobu, Akiko Takahashi, Masakazu Ito and G. Faggianelli. "Enhancing Fairness and Efficiency in PV Energy Curtailment: The Role of East–West-Facing Bifacial Installations in Radial Distribution Networks." *Energies* (2025). [https://doi.org/10.3390/en18102630](https://doi.org/10.3390/en18102630)

Moirangthem Tiken Singh, Adnan Arif, Rabinder Kumar Prasad, Bikramjit Choudhury, Chandan Kalita and S. M. S. Askari. "Optimizing Fairness and Spectral Efficiency With Shapley-Based User Prioritization in Semantic Communication." *IEEE Access*, 13 (2025): 61299-61311. [https://doi.org/10.1109/access.2025.3558107](https://doi.org/10.1109/access.2025.3558107)

Lixia Zhou, Dawei Huo, Jian Chen, Bo Bo and Hao Li. "Federated reinforcement learning with constrained markov decision processes and graph neural networks for fair and grid-constrained coordination of large-scale electric vehicle charging networks." *Scientific Reports*, 15 (2025). [https://doi.org/10.1038/s41598-025-22482-5](https://doi.org/10.1038/s41598-025-22482-5)

M. Köppen, K. Ohnishi and M. Tsuru. "Multi-Jain Fairness Index of Per-Entity Allocation Features for Fair and Efficient Allocation of Network Resources." *2013 5th International Conference on Intelligent Networking and Collaborative Systems* (2013): 841-846. [https://doi.org/10.1109/incos.2013.161](https://doi.org/10.1109/incos.2013.161)

Rahul K. Gupta and Daniel K. Molzahn. "Improving Fairness in Photovoltaic Curtailments via Daily Topology Reconfiguration for Voltage Control in Power Distribution Networks." *ArXiv*, abs/2403.07853 (2024). [https://doi.org/10.1016/j.apenergy.2025.126543](https://doi.org/10.1016/j.apenergy.2025.126543)

Yara AbdulSamadElSkaff, Hugo Joudrier, Y. Gaoua and Quoc Tuan Tran. "Towards a Fair Disaggregation in Hierarchical Cloud – Edge Control of Distributed Local Energy Communities." *2025 IEEE Kiel PowerTech* (2025): 1-6. [https://doi.org/10.1109/powertech59965.2025.11180730](https://doi.org/10.1109/powertech59965.2025.11180730)

A. B. Sediq, R. Gohary, R. Schoenen and H. Yanikomeroglu. "Optimal Tradeoff Between Sum-Rate Efficiency and Jain's Fairness Index in Resource Allocation." *IEEE Transactions on Wireless Communications*, 12 (2013): 3496-3509. [https://doi.org/10.1109/twc.2013.061413.121703](https://doi.org/10.1109/twc.2013.061413.121703)

T. Lan, M. Chiang and George Washington. "An Axiomatic Theory of Fairness in Resource Allocation." \*\* (2010).

Erhu He, Yiqun Xie, Weiye Chen, Sergii Skakun, Han Bao, Rahul Ghosh, Praveen Ravirathinam and Xiaowei Jia. "Learning With Location-Based Fairness: A Statistically-Robust Framework and Acceleration." *IEEE Transactions on Knowledge and Data Engineering*, 36 (2024): 4750-4765. [https://doi.org/10.1109/tkde.2024.3371460](https://doi.org/10.1109/tkde.2024.3371460)

A. B. Sediq, R. Gohary and H. Yanikomeroglu. "Optimal tradeoff between efficiency and Jain's fairness index in resource allocation." *2012 IEEE 23rd International Symposium on Personal, Indoor and Mobile Radio Communications \- (PIMRC)* (2012): 577-583. [https://doi.org/10.1109/pimrc.2012.6362851](https://doi.org/10.1109/pimrc.2012.6362851)

T. Hossfeld, Lea Skorin-Kapov, P. Heegaard and M. Varela. "Definition of QoE Fairness in Shared Systems." *IEEE Communications Letters*, 21 (2016): 184-187. [https://doi.org/10.1109/lcomm.2016.2616342](https://doi.org/10.1109/lcomm.2016.2616342)

V. Blanco and Ricardo G'azquez. "Fairness in maximal covering location problems." *Comput. Oper. Res.*, 157 (2022): 106287\. [https://doi.org/10.1016/j.cor.2023.106287](https://doi.org/10.1016/j.cor.2023.106287)

Arif Dataesatu, Atsushi Wakayama, Kazuo Ibuka, Homare Murakami and Takeshi Matsumura. "Fair Resource Allocation with Dynamic Multi-Layer Service Area Management in Local Networks Supporting Cybernetic Avatars." *2025 IEEE 101st Vehicular Technology Conference (VTC2025-Spring)* (2025): 1-7. [https://doi.org/10.1109/vtc2025-spring65109.2025.11174886](https://doi.org/10.1109/vtc2025-spring65109.2025.11174886)

A. K. Zarabie, Sanjoy Das and M. N. Faqiry. "Fairness-Regularized DLMP-Based Bilevel Transactive Energy Mechanism in Distribution Systems." *IEEE Transactions on Smart Grid*, 10 (2018): 6029-6040. [https://doi.org/10.1109/tsg.2019.2895527](https://doi.org/10.1109/tsg.2019.2895527)

Saeedeh Ghazi, Saeed Farzi and A. Nikoofard. "Federated Learning for All: A Reinforcement Learning-Based Approach for Ensuring Fairness in Client Selection." *IEEE Access*, 13 (2025): 118515-118535. [https://doi.org/10.1109/access.2025.3586943](https://doi.org/10.1109/access.2025.3586943)

M. Dynge and Umit Cali. "Distributive energy justice in local electricity markets: Assessing the performance of fairness indicators." *Applied Energy* (2025). [https://doi.org/10.1016/j.apenergy.2025.125463](https://doi.org/10.1016/j.apenergy.2025.125463)

Rubén González-Sendino, Emilio Serrano and Javier Bajo. "Quantifying algorithmic discrimination: A two-dimensional approach to fairness in artificial intelligence." *Eng. Appl. Artif. Intell.*, 144 (2025): 109979\. [https://doi.org/10.1016/j.engappai.2024.109979](https://doi.org/10.1016/j.engappai.2024.109979)

T. Lan, D. Kao, M. Chiang and A. Sabharwal. "An Axiomatic Theory of Fairness." *ArXiv*, abs/0906.0557 (2009).

N. Mehmood and N. Arshad. "Towards Developing a Large Distributed Energy Storage Using a Weighted Batteries Scheduling Scheme." *IEEE Access*, 8 (2020): 210733-210749. [https://doi.org/10.1109/access.2020.3039924](https://doi.org/10.1109/access.2020.3039924)

Rahul K. Gupta, Paprapee Buason and D. Molzahn. "Fairness-Aware Photovoltaic Generation Limits for Voltage Regulation in Power Distribution Networks Using Conservative Linear Approximations." *2024 IEEE Texas Power and Energy Conference (TPEC)* (2024): 1-6. [https://doi.org/10.1109/tpec60005.2024.10472277](https://doi.org/10.1109/tpec60005.2024.10472277)

C. Fraunholz, Ali Tash, Heike Scheben and Alexander Zillich. "Demand curtailment allocation in interconnected electricity markets." *Applied Energy* (2025). [https://doi.org/10.1016/j.apenergy.2024.124679](https://doi.org/10.1016/j.apenergy.2024.124679)

A. Wismadi, M. Maarseveen and M. Brussel. "Spatial and social equity index for equity based resource allocation." \*\* (2009).

\---

# **Jain’s Fairness Index is a useful but limited metric for resource distribution in multi-agent competitive games.**

## **1\. Introduction**

Jain’s Fairness Index (JFI) is widely used to quantify equity in resource allocation among agents, especially in networking, edge computing, and multi-agent systems (Jain et al. 1998; Lan et al. 2009, 1-9; Cheraghy et al. 2024, 1-5; Köppen et al. 2013, 841-846). Its popularity stems from its simplicity, scale invariance, and ability to summarize fairness as a single value between 0 (most unfair) and 1 (perfectly fair) (Jain et al. 1998; Lan et al. 2009, 1-9). Empirical studies show that JFI aligns well with human perceptions of fairness in one-to-many allocation games (Grappiolo et al. 2013, 176-200), and it is frequently used to benchmark or optimize fairness in competitive environments such as multi-player bandit problems and wireless networks (Leng 2025; Cheraghy et al. 2024, 1-5; Kumar and Yeoh 2025, 2591-2593; Ghazi et al. 2025, 118515-118535). However, several papers highlight limitations: JFI does not account for context, agent needs, or protected attributes, and may fail to capture group or procedural fairness (Fossati et al. 2017, 1-9; Bertsimas et al. 2011, 17-31; Köppen et al. 2013, 841-846). Extensions like the Multi-Jain index and game-theoretic “mood value” have been proposed to address these gaps (Fossati et al. 2017, 1-9; Köppen et al. 2013, 841-846). Overall, JFI is effective as a baseline metric but should be complemented by more nuanced measures in complex or strategic settings (Lan et al. 2009, 1-9; Fossati et al. 2017, 1-9; Bertsimas et al. 2011, 17-31; Köppen et al. 2013, 841-846).

**Figure 1:** Consensus on the effectiveness of Jain's Fairness Index for multi-agent resource allocation.

## **2\. Methods**

A comprehensive search was conducted across over 170 million research papers using Consensus, which aggregates sources like Semantic Scholar and PubMed. The search identified 948 potentially relevant papers, screened 506 after de-duplication, filtered 398 for eligibility based on relevance to Jain’s Fairness Index in multi-agent competitive games, and included the top 50 most pertinent papers for this review.

| Identification | Screening | Eligibility | Included |
| ----- | ----- | ----- | ----- |
| 948 | 506 | 398 | 50 |

**Figure 2:** Flow diagram of paper selection process for this review.

Twenty unique searches were executed across eight search groups targeting foundational theory, empirical validation, alternative metrics, group fairness, efficiency–fairness trade-offs, and critiques of JFI.

## **3\. Results**

### **3.1 Adoption and Strengths of Jain’s Fairness Index**

JFI is extensively adopted due to its ease of computation and interpretability. It is used as a standard benchmark for evaluating fairness in resource allocation algorithms across networking (Jain et al. 1998), edge computing (Leng 2025), federated learning (Ghazi et al. 2025, 118515-118535), and cooperative/competitive game scenarios (Cheraghy et al. 2024, 1-5; Kumar and Yeoh 2025, 2591-2593). Studies confirm that maximizing JFI often leads to more equitable outcomes without significant loss of efficiency (Cheraghy et al. 2024, 1-5; Sediq et al. 2012, 577-583).

### **3.2 Empirical Validation and Human Alignment**

Empirical work demonstrates that JFI correlates well with human judgments of fairness in game-based resource allocation tasks. In one study using crowdsourced annotations from a computer game scenario, JFI was among the most expressive metrics for capturing perceived fairness alongside normalized entropy (Grappiolo et al. 2013, 176-200).

### **3.3 Limitations and Critiques**

Several papers critique JFI’s limitations:

* **Context Insensitivity:** JFI does not consider agent needs or protected attributes; it only evaluates the final distribution vector (Fossati et al. 2017, 1-9; Bertsimas et al. 2011, 17-31).  
* **Single-Metric Blindness:** In multi-dimensional or strategic settings (e.g., when multiple resources or features are involved), applying JFI to a single observable can miss important aspects like proportionality or envy-freeness; extensions like Multi-Jain indices are proposed to address this (Köppen et al. 2013, 841-846).  
* **Not Group-Fair:** Group-based or algorithmic fairness notions (e.g., demographic parity) are not captured by JFI (Bertsimas et al. 2011, 17-31).

### **3.4 Extensions and Alternatives**

Extensions such as the “mood value” adapt JFI by incorporating individual satisfaction rates within cooperative game theory frameworks (Fossati et al. 2017, 1-9; Fossati et al. 2018, 2801-2814). Multi-metric approaches apply feature-wise Jain indices to better represent justness across multiple dimensions (Köppen et al. 2013, 841-846). Other works propose integrating Shapley values or welfare-based dominance constraints for richer fairness analysis (Singh et al. 2025, 61299-61311; Argyris et al. 2021, 560-578).

#### **Results Timeline**

* **2009**  
  * 1 paper: (Lan et al. 2009, 1-9)- **2010**  
  * 1 paper: (Lan et al. 2010)- **2011**  
  * 1 paper: (Bertsimas et al. 2011, 17-31)- **2012**  
  * 1 paper: (Sediq et al. 2012, 577-583)- **2013**  
  * 1 paper: (Grappiolo et al. 2013, 176-200)- **2015**  
  * 1 paper: (Nicosia et al. 2015)- **2017**  
  * 1 paper: (Fossati et al. 2017, 1-9)- **2022**  
  * 1 paper: (Salem et al. 2022, 1 \- 36)- **2024**  
  * 4 papers: (Leng 2025; Reddy 2024; Cheraghy et al. 2024, 1-5; Chen et al. 2024)- **2025**  
  * 8 papers: (Li and Duan 2025, 8393-8404; Binkyte 2025; Kumar and Yeoh 2025, 2591-2593; Gharbi et al. 2025; Zargari et al. 2025, 1 \- 51; Tluczek et al. 2025; Chen et al. 2025, 498-512; Ghazi et al. 2025, 118515-118535)**Figure 3:** Timeline showing evolution of research on Jain's Fairness Index in multi-agent systems. Larger markers indicate more citations.

#### **Top Contributors**

| Type | Name | Papers |
| ----- | ----- | ----- |
| Author | T. Lan | (Lan et al. 2009, 1-9; Lan et al. 2010\) |
| Author | Francesca Fossati | (Fossati et al. 2017, 1-9; Fossati et al. 2018, 2801-2814) |
| Author | A. B. Sediq | (Sediq et al. 2012, 577-583; Sediq et al. 2013, 3496-3509) |
| Journal | *IEEE Access* | (Ghazi et al. 2025, 118515-118535; Singh et al. 2025, 61299-61311) |
| Journal | *Proceedings of the ACM on Measurement and Analysis of Computing Systems* | (Salem et al. 2022, 1 \- 36; Zargari et al. 2025, 1 \- 51; Liu and Hajiesmaili 2025, 1 \- 46\) |
| Journal | *ArXiv* | (Nicosia et al. 2015; Kumar and Yeoh 2025; Filter et al. 2025\) |

**Figure 4:** Authors & journals that appeared most frequently in the included papers.

## **4\. Discussion**

Jain’s Fairness Index remains a foundational tool for quantifying equity in resource distribution among agents due to its simplicity and broad applicability (Jain et al. 1998; Lan et al. 2009, 1-9). Its empirical alignment with human perceptions makes it valuable for benchmarking algorithms where distributive justice is central (Grappiolo et al. 2013, 176-200). However, its limitations—especially regarding context insensitivity and inability to capture group or procedural fairness—are significant when applied to complex multi-agent competitive games where agent heterogeneity or protected attributes matter (Fossati et al. 2017, 1-9; Bertsimas et al. 2011, 17-31; Köppen et al. 2013, 841-846).

Recent research advocates supplementing JFI with context-aware metrics (e.g., need-based allocations), group-fairness measures (e.g., demographic parity), or satisfaction-based indices (“mood value”) for richer analysis (Fossati et al. 2017, 1-9; Bertsimas et al. 2011, 17-31; Fossati et al. 2018, 2801-2814). Multi-metric approaches like Multi-Jain indices provide a more nuanced view when multiple resources/features are at play (Köppen et al. 2013, 841-846). Theoretical work further grounds JFI within broader families of axiomatic fairness measures but also highlights cases where it may be insufficient alone (Lan et al. 2009, 1-9).

### **Claims and Evidence Table**

| Claim | Evidence Strength | Reasoning | Papers |
| ----- | ----- | ----- | ----- |
| Jain’s Fairness Index is widely adopted as a baseline equity metric | Evidence strength: Strong (9/10) | Extensively cited; simple; interpretable; used across networking/MAS/game theory | (Jain et al. 1998; Lan et al. 2009, 1-9; Cheraghy et al. 2024, 1-5) |
| JFI aligns well with human perceptions of fairness | Evidence strength: Strong (8/10) | Empirical studies show strong correlation with crowdsourced human ratings | (Grappiolo et al. 2013, 176-200) |
| JFI fails to capture context-specific/group/procedural fairness | Evidence strength: Moderate (7/10) | Lacks sensitivity to needs/attributes; cannot express group discrimination | (Fossati et al. 2017, 1-9; Bertsimas et al. 2011, 17-31) |
| Multi-metric extensions improve upon single-metric JFI | Evidence strength: Moderate (6/10) | Feature-wise indices better capture proportionality/envy-freeness/equity | (Köppen et al. 2013, 841-846) |
| Game-theoretic “mood value” addresses some limitations | Evidence strength: Moderate (5/10) | Incorporates satisfaction rates; adapts index for cooperative/strategic settings | (Fossati et al. 2017, 1-9; Fossati et al. 2018, 2801-2814) |
| Using only JFI can be misleading in complex/heterogeneous settings | Evidence strength: Moderate (4/10) | May mask unfair outcomes if needs/context ignored | (Bertsimas et al. 2011, 17-31) |

**Figure undefined:** Key claims and support evidence identified in these papers.

## **5\. Conclusion**

Jain’s Fairness Index is an effective baseline metric for measuring equity in resource distribution among agents in competitive games but has notable limitations regarding context sensitivity and group/procedural fairness. It should be complemented by additional metrics tailored to specific application requirements.

### **Research Gaps**

Despite extensive use of JFI as a scalar measure of distributive justice, there are gaps regarding its application to heterogeneous agent populations (with varying needs/rights), multidimensional resources/features, dynamic/adaptive environments, and integration with group/procedural fairness concepts.

#### **Research Gaps Matrix**

| Topic/Outcome | Homogeneous Agents | Heterogeneous Agents | Multi-Dimensional Resources | Group Fairness Metrics |
| ----- | ----- | ----- | ----- | ----- |
| Scalar Equity (JFI) | **18** | **5** | **4** | **GAP** |
| Context-Aware Fairness | **1** | **7** | **2** | **4** |
| Group/Protected Attributes | **GAP** | **GAP** | **GAP** | **5** |
| Satisfaction-Based Indices | **GAP** | **2** | **1** | **GAP** |

**Figure undefined:** Matrix showing coverage of topics versus study attributes; zeros indicate underexplored areas.

### **Open Research Questions**

Future research should focus on integrating context-aware/group-aware metrics with scalar indices like JFI; developing robust benchmarks for heterogeneous/multi-dimensional settings; and empirically validating new metrics against human perceptions.

| Question | Why |
| ----- | ----- |
| **How can Jain's Fairness Index be extended to account for agent heterogeneity or protected attributes?** | Addressing this would enable fairer allocations reflecting real-world diversity beyond scalar equity. |
| **What combinations of scalar (JFI) and group/contextual metrics best align with human perceptions of fairness?** | Understanding this could guide practical algorithm design balancing interpretability with just outcomes. |
| **How do dynamic/adaptive environments affect the reliability of scalar fairness indices?** | Exploring this would clarify whether static indices remain valid under changing conditions. |

**Figure undefined:** Open questions highlight directions for advancing fair resource allocation metrics.

In summary: Jain’s Fairness Index is an effective starting point but should be supplemented by richer metrics tailored to specific contexts within multi-agent competitive games.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Xiutong Leng. "Strategic Learning in Multi-Player Bandit Problems: A Game-Theoretic Approach to Resource Allocation in Edge Computing." *ITM Web of Conferences* (2025). [https://doi.org/10.1051/itmconf/20257803027](https://doi.org/10.1051/itmconf/20257803027)

T. Lan, D. Kao, M. Chiang and A. Sabharwal. "An Axiomatic Theory of Fairness in Network Resource Allocation." *2010 Proceedings IEEE INFOCOM* (2009): 1-9. [https://doi.org/10.1109/infcom.2010.5461911](https://doi.org/10.1109/infcom.2010.5461911)

Francesca Fossati, Stefano Moretti and Stefano Secci. "A mood value for fair resource allocations." *2017 IFIP Networking Conference (IFIP Networking) and Workshops* (2017): 1-9. [https://doi.org/10.23919/ifipnetworking.2017.8264839](https://doi.org/10.23919/ifipnetworking.2017.8264839)

Katha Rohan Reddy. "Achieving Fairness with Intelligent Co Agents." *Indian Journal of Artificial Intelligence and Neural Networking* (2024). [https://doi.org/10.54105/ijainn.a1080.04011223](https://doi.org/10.54105/ijainn.a1080.04011223)

T. Si Salem, G. Iosifidis and G. Neglia. "Enabling Long-term Fairness in Dynamic Resource Allocation." *Proceedings of the ACM on Measurement and Analysis of Computing Systems*, 6 (2022): 1 \- 36\. [https://doi.org/10.1145/3570606](https://doi.org/10.1145/3570606)

D. Bertsimas, V. Farias and Nikolaos Trichakis. "The Price of Fairness." *Oper. Res.*, 59 (2011): 17-31. [https://doi.org/10.1287/opre.1100.0865](https://doi.org/10.1287/opre.1100.0865)

Maryam Cheraghy, Meysam Soltanpour, Belal Abuhaija, H. B. Abdalla and Kennedy Ehimwenma. "Game-Theoretic-Based Resource Allocation Algorithm for SCMA Max-Min Problem to Maximize Fairness." *2024 IEEE 7th International Conference on Electronics and Communication Engineering (ICECE)* (2024): 1-5. [https://doi.org/10.1109/icece63871.2024.10976939](https://doi.org/10.1109/icece63871.2024.10976939)

A. B. Sediq, R. Gohary and H. Yanikomeroglu. "Optimal tradeoff between efficiency and Jain's fairness index in resource allocation." *2012 IEEE 23rd International Symposium on Personal, Indoor and Mobile Radio Communications \- (PIMRC)* (2012): 577-583. [https://doi.org/10.1109/pimrc.2012.6362851](https://doi.org/10.1109/pimrc.2012.6362851)

T. Lan, M. Chiang and George Washington. "An Axiomatic Theory of Fairness in Resource Allocation." \*\* (2010).

G. Nicosia, A. Pacifici and U. Pferschy. "Price of Fairness for allocating a bounded resource." *ArXiv*, abs/1508.05253 (2015). [https://doi.org/10.1016/j.ejor.2016.08.013](https://doi.org/10.1016/j.ejor.2016.08.013)

Hongbo Li and Lingjie Duan. "Competitive Multi-Armed Bandit Games for Resource Sharing." *IEEE Transactions on Mobile Computing*, 24 (2025): 8393-8404. [https://doi.org/10.1109/tmc.2025.3555971](https://doi.org/10.1109/tmc.2025.3555971)

Ruta Binkyte. "Interactional Fairness in LLM Multi-Agent Systems: An Evaluation Framework." *ArXiv*, abs/2505.12001 (2025). [https://doi.org/10.48550/arxiv.2505.12001](https://doi.org/10.48550/arxiv.2505.12001)

Ashwin Kumar and William Yeoh. "DECAF: Learning to be Fair in Multi-agent Resource Allocation." \*\* (2025): 2591-2593. [https://doi.org/10.48550/arxiv.2502.04281](https://doi.org/10.48550/arxiv.2502.04281)

Qian Chen, Kanyu Bao, Yulin Wu, Xiaozhen Sun, Xuan Wang, Z. L. Jiang, Shuhan Qi, Yifan Li and Lei Cui. "Fairness mechanism and stackelberg strategy for multi-agent systems: A case study of legends of the three kingdoms." *Alexandria Engineering Journal* (2024). [https://doi.org/10.1016/j.aej.2024.07.125](https://doi.org/10.1016/j.aej.2024.07.125)

Atef Gharbi, Mohamed Ayari, Nasser Albalawi, Yamen El Touati and Zeineb Klai. "A Comparative Analysis of Fairness and Satisfaction in Multi-Agent Resource Allocation: Integrating Borda Count and K-Means Approaches with Distributive Justice Principles." *Mathematics* (2025). [https://doi.org/10.3390/math13152355](https://doi.org/10.3390/math13152355)

Faraz Zargari, Hossein Nekouyan, Bo Sun and Xiaoqi Tan. "Online Allocation with Multi-Class Arrivals: Group Fairness vs Individual Welfare." *Proceedings of the ACM on Measurement and Analysis of Computing Systems*, 9 (2025): 1 \- 51\. [https://doi.org/10.1145/3727120](https://doi.org/10.1145/3727120)

Jakub Tluczek, Victor Villin and Christos Dimitrakakis. "Fair Contracts in Principal-Agent Games with Heterogeneous Types." *ArXiv*, abs/2506.15887 (2025). [https://doi.org/10.48550/arxiv.2506.15887](https://doi.org/10.48550/arxiv.2506.15887)

Geng Chen, Lili Cheng, Qingtian Zeng, Fei Shen and Yu-dong Zhang. "Willingness Allocation-Assisted Cooperative Localization Algorithm Based on Competitive Game for Resource-Constrained Environment." *IEEE Transactions on Green Communications and Networking*, 9 (2025): 498-512. [https://doi.org/10.1109/tgcn.2024.3436535](https://doi.org/10.1109/tgcn.2024.3436535)

Saeedeh Ghazi, Saeed Farzi and A. Nikoofard. "Federated Learning for All: A Reinforcement Learning-Based Approach for Ensuring Fairness in Client Selection." *IEEE Access*, 13 (2025): 118515-118535. [https://doi.org/10.1109/access.2025.3586943](https://doi.org/10.1109/access.2025.3586943)

Corrado Grappiolo, H. P. Martínez and Georgios N. Yannakakis. "Validating Generic Metrics of Fairness in Game-Based Resource Allocation Scenarios with Crowdsourced Annotations." *Trans. Comput. Collect. Intell.*, 13 (2013): 176-200. [https://doi.org/10.1007/978-3-642-54455-2\_8](https://doi.org/10.1007/978-3-642-54455-2_8)

Ashwin Kumar and William Yeoh. "A General Incentives-Based Framework for Fairness in Multi-agent Resource Allocation." *ArXiv*, abs/2510.26740 (2025). [https://doi.org/10.48550/arxiv.2510.26740](https://doi.org/10.48550/arxiv.2510.26740)

Moirangthem Tiken Singh, Adnan Arif, Rabinder Kumar Prasad, Bikramjit Choudhury, Chandan Kalita and S. M. S. Askari. "Optimizing Fairness and Spectral Efficiency With Shapley-Based User Prioritization in Semantic Communication." *IEEE Access*, 13 (2025): 61299-61311. [https://doi.org/10.1109/access.2025.3558107](https://doi.org/10.1109/access.2025.3558107)

Björn Filter, Ralf Möller and Ö. Özçep. "A Mechanism for Mutual Fairness in Cooperative Games with Replicable Resources \- Extended Version." *ArXiv*, abs/2508.13960 (2025). [https://doi.org/10.48550/arxiv.2508.13960](https://doi.org/10.48550/arxiv.2508.13960)

M. Köppen, K. Ohnishi and M. Tsuru. "Multi-Jain Fairness Index of Per-Entity Allocation Features for Fair and Efficient Allocation of Network Resources." *2013 5th International Conference on Intelligent Networking and Collaborative Systems* (2013): 841-846. [https://doi.org/10.1109/incos.2013.161](https://doi.org/10.1109/incos.2013.161)

A. B. Sediq, R. Gohary, R. Schoenen and H. Yanikomeroglu. "Optimal Tradeoff Between Sum-Rate Efficiency and Jain's Fairness Index in Resource Allocation." *IEEE Transactions on Wireless Communications*, 12 (2013): 3496-3509. [https://doi.org/10.1109/twc.2013.061413.121703](https://doi.org/10.1109/twc.2013.061413.121703)

R. Jain, D. Chiu and W. Hawe. "A Quantitative Measure Of Fairness And Discrimination For Resource Allocation In Shared Computer Systems." *ArXiv*, cs.NI/9809099 (1998).

Qingsong Liu and M. Hajiesmaili. "Online Fair Allocation of Reusable Resources." *Proceedings of the ACM on Measurement and Analysis of Computing Systems*, 9 (2025): 1 \- 46\. [https://doi.org/10.1145/3727121](https://doi.org/10.1145/3727121)

Francesca Fossati, Sahar Hoteit, Stefano Moretti and Stefano Secci. "Fair Resource Allocation in Systems With Complete Information Sharing." *IEEE/ACM Transactions on Networking*, 26 (2018): 2801-2814. [https://doi.org/10.1109/tnet.2018.2878644](https://doi.org/10.1109/tnet.2018.2878644)

N. Argyris, Özlem Karsu and M. Yavuz. "Fair resource allocation: Using welfare-based dominance constraints." *Eur. J. Oper. Res.*, 297 (2021): 560-578. [https://doi.org/10.1016/j.ejor.2021.05.003](https://doi.org/10.1016/j.ejor.2021.05.003)

\---

# **Evidence is insufficient to link Moran’s I in map design to player perceived fairness**

## **1\. Introduction**

Spatial autocorrelation, commonly measured by **Moran’s I**, is a foundational concept in spatial statistics and is widely used to analyze clustering or dispersion in spatial data (Chen 2022; Chen 2020). In map and level design—especially for games—fairness is a critical aspect of player experience, but the relationship between the spatial structure of maps (as quantified by Moran’s I) and **player-perceived fairness** remains largely unexplored. While some research addresses fairness in spatial resource allocation or urban planning using spatial autocorrelation metrics (Cheng et al. 2022; Liu et al. 2021; Whitehead et al. 2019, 270 \- 278; Jian et al. 2020, 102122), there is no direct evidence that higher or lower Moran’s I values in game maps correlate with how fair players perceive those maps to be. Studies on procedural content generation and game design highlight the importance of balance and playability but do not empirically connect these qualities to spatial autocorrelation indices (Silva et al. 2025, 34179 \- 34205; Risi and Togelius 2019, 428 \- 436; Silva et al. 2024, 345-364). Similarly, research on subjective perceptions of space often focuses on attributes like accessibility, aesthetics, or safety rather than statistical measures like Moran’s I (Kaklauskas et al. 2021, 105458; Ogawa et al. 2024; Huang et al. 2025). Thus, while both concepts are well-studied independently, their intersection—specifically regarding player fairness perception—remains an open question.

**Figure 1:** Consensus on Moran’s I–fairness link in games

## **2\. Methods**

A comprehensive search was conducted across over 170 million research papers indexed by Consensus, including Semantic Scholar, PubMed, and other sources. The search strategy involved 22 unique queries spanning foundational theory, map design applications, fairness perception, adjacent fields (urban planning, HCI), and critiques of spatial autocorrelation for subjective assessment. In total, 1124 papers were identified; after screening and eligibility filtering, 454 were deemed relevant. The top 20 most pertinent papers were included in this review.

| Identification | Screening | Eligibility | Included |
| ----- | ----- | ----- | ----- |
| 1124 | 735 | 454 | 20 |

**Figure 2:** Flowchart of paper selection process for this review.

This review synthesized findings from 22 unique searches covering both technical and perceptual aspects of spatial autocorrelation and fairness.

## **3\. Results**

### **3.1 Spatial Autocorrelation: Theory and Application**

Moran’s I is extensively used to quantify global or local clustering/dispersion patterns in spatial data (Chen 2022; Chen 2020). Its mathematical properties are well-understood and it has been applied to urban systems, land cover mapping, transportation networks, and health service equity (Chen 2022; Cheng et al. 2022; Liu et al. 2021; Chen 2020; Whitehead et al. 2019, 270 \- 278; He et al. 2023).

### **3.2 Fairness: Definitions and Measurement**

Fairness in spatial contexts is often defined as **spatial equity** or **justice**, focusing on the distribution of resources or accessibility rather than subjective perceptions (Cheng et al. 2022; Liu et al. 2021; Whitehead et al. 2019, 270 \- 278; Amorim et al. 2025; Jian et al. 2020, 102122). These studies use measures like Moran’s I to assess whether services or amenities are equitably distributed but do not address user perceptions directly.

### **3.3 Game Design: Procedural Generation & Balance**

Procedural content generation (PCG) research emphasizes the need for balanced and playable levels but rarely quantifies balance using spatial autocorrelation metrics (Silva et al. 2025, 34179 \- 34205; Risi and Togelius 2019, 428 \- 436; Silva et al. 2024, 345-364). Calls for user studies highlight a gap between algorithmic measures (like Moran's I) and actual player experience (Silva et al. 2025, 34179 \- 34205).

### **3.4 Subjective Perception: Urban & Game Environments**

Studies measuring subjective perceptions (e.g., attractiveness, safety) use surveys or machine learning models based on visual features rather than statistical indices like Moran's I (Kaklauskas et al. 2021, 105458; Ogawa et al. 2024; Huang et al. 2025). No empirical studies were found linking map-level Moran's I values to perceived fairness among users or players.

#### **Results Timeline**

* **Aug 2019**  
  * 1 paper: (Whitehead et al. 2019, 270 \- 278)- **Nov 2019**  
  * 1 paper: (Risi and Togelius 2019, 428 \- 436)- **Jan 2020**  
  * 1 paper: (Chen 2020)- **Feb 2020**  
  * 1 paper: (Jian et al. 2020, 102122)- **Apr 2021**  
  * 1 paper: (Markello and Mišić 2021)- **Jun 2021**  
  * 1 paper: (Flynn et al. 2021)- **Jul 2021**  
  * 1 paper: (Kaklauskas et al. 2021, 105458)- **Sep 2021**  
  * 1 paper: (Liu et al. 2021)- **Aug 2022**  
  * 1 paper: (Cheng et al. 2022)- **Sep 2022**  
  * 1 paper: (Chen 2022)- **Nov 2022**  
  * 1 paper: (Caroux 2022, 103936 )- **Jan 2023**  
  * 1 paper: (Kosch et al. 2023, 1 \- 39)- **Mar 2023**  
  * 1 paper: (Putra et al. 2023)- **Nov 2023**  
  * 1 paper: (He et al. 2023)- **Jun 2024**  
  * 1 paper: (Ogawa et al. 2024)- **Aug 2024**  
  * 1 paper: (Silva et al. 2024, 345-364)- **Jan 2025**  
  * 1 paper: (Silva et al. 2025, 34179 \- 34205)- **Apr 2025**  
  * 1 paper: (Huang et al. 2025)- **May 2025**  
  * 1 paper: (Amorim et al. 2025)- **Jun 2025**  
  * 1 paper: (Xue et al. 2025\)**Figure 3:** Timeline of key publications on spatial autocorrelation, fairness metrics, and perception studies. Larger markers indicate more citations.

#### **Top Contributors**

| Type | Name | Papers |
| ----- | ----- | ----- |
| Author | Yanguang Chen | (Chen 2022; Chen 2020\) |
| Author | Gang Cheng | (Cheng et al. 2022\) |
| Author | Bingxi Liu | (Liu et al. 2021\) |
| Journal | *Sustainability* | (Cheng et al. 2022; Huang et al. 2025\) |
| Journal | *Scientific Reports* | (Chen 2022\) |
| Journal | *PLoS ONE* | (Chen 2020\) |

**Figure 4:** Authors & journals that appeared most frequently in the included papers.

## **4\. Discussion**

The literature demonstrates robust methods for measuring spatial autocorrelation (Moran's I) and formalizing various notions of fairness or equity in resource distribution (Chen 2022; Cheng et al. 2022; Liu et al. 2021; Chen 2020; Whitehead et al. 2019, 270 \- 278). However, there is a clear disconnect between these quantitative measures and **subjective player perceptions** of fairness in game environments. While some works propose frameworks for evaluating user perceptions of space quality or emotional response (Kaklauskas et al. 2021, 105458; Ogawa et al. 2024), none empirically test whether changes in a map's Moran's I value affect how fair players perceive it to be.

Research on procedural content generation acknowledges the importance of playability and balance but does not operationalize these concepts using spatial statistics like Moran's I (Silva et al. 2025, 34179 \- 34205; Risi and Togelius 2019, 428 \- 436). Similarly, urban planning literature uses spatial autocorrelation to diagnose inequity but stops short of linking these patterns to lived experiences or perceived justice among residents (Cheng et al. 2022; Liu et al. 2021).

Overall, while both domains are mature independently—spatial statistics for pattern analysis; fairness/equity for resource distribution—their intersection as it relates to **player-perceived fairness** remains untested.

### **Claims and Evidence Table**

| Claim | Evidence Strength | Reasoning | Papers |
| ----- | ----- | ----- | ----- |
| Moran’s I effectively quantifies global clustering/dispersion | Evidence strength: Strong (10/10) | Widely validated across disciplines; standard tool for detecting non-random patterns | (Chen 2022; Chen 2020\) |
| Spatial equity/fairness can be assessed using spatial statistics | Evidence strength: Strong (8/10) | Used extensively in urban planning/transportation/resource allocation studies | (Cheng et al. 2022; Liu et al. 2021; Whitehead et al. 2019, 270 \- 278\) |
| No direct evidence links map-level Moran’s I to player-perceived fairness | Evidence strength: Strong (8/10) | Comprehensive search found no empirical studies testing this relationship | (Silva et al. 2025, 34179 \- 34205; Kaklauskas et al. 2021, 105458; Ogawa et al. 2024\) |
| Subjective perceptions are influenced by visual/aesthetic factors more than statistical indices | Evidence strength: Moderate (6/10) | Studies use surveys/ML models based on images rather than statistical measures | (Kaklauskas et al. 2021, 105458; Ogawa et al. 2024\) |
| Procedural content generation research calls for user studies connecting algorithmic metrics with experience | Evidence strength: Moderate (5/10) | Gaps identified; future work suggested but not yet realized | (Silva et al. 2025, 34179 \- 34205\) |
| Urban planning literature lacks consensus on definitions/measures of spatial equity/fairness | Evidence strength: Moderate (4/10) | Systematic reviews note definitional/methodological diversity; little focus on perception | (Whitehead et al. 2019, 270 \- 278\) |

**Figure undefined:** Key claims and support evidence identified in these papers.

## **5\. Conclusion**

Current research provides strong tools for measuring both **spatial autocorrelation** (Moran's I) and various forms of **spatial equity/fairness**, but there is no empirical evidence that directly connects these measures with **player-perceived fairness** in map design. This represents a significant gap at the intersection of quantitative analysis and user experience.

### **Research Gaps**

| Topic/Outcome | Game Maps | Urban Planning | User Perception | Procedural Generation |
| ----- | ----- | ----- | ----- | ----- |
| Use of Moran's I | **GAP** | **4** | **GAP** | **1** |
| Fairness/Equity Assessment | **GAP** | **5** | **1** | **1** |
| Player/User Perceived Fairness | **GAP** | **GAP** | **2** | **GAP** |

**Figure undefined:** Matrix showing where research exists versus gaps at the intersection of topics.

### **Open Research Questions**

Future work should empirically test whether manipulating a map's global/local Moran's I affects player judgments about fairness.

| Question | Why |
| ----- | ----- |
| **Does varying the global/local Moran’s I value of a game map influence players’ perceived fairness?** | Directly tests if statistical clustering/dispersion impacts subjective experience—a missing link between theory & practice. |
| **What visual/spatial features most strongly predict perceived fairness compared to statistical indices?** | Identifies which aspects matter most for users—guiding designers toward effective interventions beyond raw statistics. |

**Figure undefined:** Open questions at the intersection of map structure metrics and user perception.

In summary: There is currently no direct evidence that higher or lower Moran’s I values in game maps correlate with player-perceived fairness—highlighting an important opportunity for future interdisciplinary research bridging quantitative analysis with human-centered evaluation.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Cheryl J. Flynn, S. Majumdar and Ritwik Mitra. "Evaluating Fairness in the Presence of Spatial Autocorrelation." *ArXiv*, abs/2101.01703 (2021).

Daniele F. Silva, R. Torchelsen and Marilton S. Aguiar. "Procedural game level generation with GANs: potential, weaknesses, and unresolved challenges in the literature." *Multimedia Tools and Applications*, 84 (2025): 34179 \- 34205\. [https://doi.org/10.1007/s11042-025-20612-9](https://doi.org/10.1007/s11042-025-20612-9)

Yanguang Chen. "Spatial autocorrelation equation based on Moran’s index." *Scientific Reports*, 13 (2022). [https://doi.org/10.1038/s41598-023-45947-x](https://doi.org/10.1038/s41598-023-45947-x)

T. Kosch, Jakob Karolus, Johannes Zagermann, Harald Reiterer, Albrecht Schmidt and Paweł W. Woźniak. "A Survey on Measuring Cognitive Workload in Human-Computer Interaction." *ACM Computing Surveys*, 55 (2023): 1 \- 39\. [https://doi.org/10.1145/3582272](https://doi.org/10.1145/3582272)

A. Kaklauskas, D. Bardauskienė, R. Cerkauskiene, I. Ubarte, Saulius Raslanas, E. Radvile, U. Kaklauskaitė and L. Kaklauskiene. "Emotions analysis in public spaces for urban planning." *Land Use Policy*, 107 (2021): 105458\. [https://doi.org/10.1016/j.landusepol.2021.105458](https://doi.org/10.1016/j.landusepol.2021.105458)

Yoshiki Ogawa, Takuya Oki, Chenbo Zhao, Y. Sekimoto and C. Shimizu. "Evaluating the subjective perceptions of streetscapes using street-view images." *Landscape and Urban Planning* (2024). [https://doi.org/10.1016/j.landurbplan.2024.105073](https://doi.org/10.1016/j.landurbplan.2024.105073)

Gang Cheng, Lei Guo and Tao Zhang. "Spatial Equity Assessment of Bus Travel Behavior for Pilgrimage: Evidence from Lhasa, Tibet, China." *Sustainability* (2022). [https://doi.org/10.3390/su141710486](https://doi.org/10.3390/su141710486)

S. Risi and J. Togelius. "Increasing generality in machine learning through procedural content generation." *Nature Machine Intelligence*, 2 (2019): 428 \- 436\. [https://doi.org/10.1038/s42256-020-0208-z](https://doi.org/10.1038/s42256-020-0208-z)

Bingxi Liu, Yunqing Tian, Mengjie Guo, D. Tran, Abdulfattah.A.Q. Alwah and Dawei Xu. "Evaluating the disparity between supply and demand of park green space using a multi-dimensional spatial equity evaluation framework." *Cities* (2021). [https://doi.org/10.1016/j.cities.2021.103484](https://doi.org/10.1016/j.cities.2021.103484)

B. C. D. Silva, J. G. R. Maia and Windson Viana. "Procedural content generation in pervasive games: state of affairs, mistakes, and successes." *Int. J. Pervasive Comput. Commun.*, 20 (2024): 345-364. [https://doi.org/10.1108/ijpcc-11-2023-0314](https://doi.org/10.1108/ijpcc-11-2023-0314)

Yanyan Huang, Lanxin Ye and Ye Chen. "Sustainable Urban Landscape Quality: A User-Perception Framework for Public Space Assessment and Development." *Sustainability* (2025). [https://doi.org/10.3390/su17093992](https://doi.org/10.3390/su17093992)

Dimas Widya Putra, W. Salim, P. N. Indradjati and Niken Prilandita. "Understanding the position of urban spatial configuration on the feeling of insecurity from crime in public spaces." \*\*, 9 (2023). [https://doi.org/10.3389/fbuil.2023.1114968](https://doi.org/10.3389/fbuil.2023.1114968)

R. Markello and B. Mišić. "Comparing spatial null models for brain maps." *NeuroImage*, 236 (2021). [https://doi.org/10.1016/j.neuroimage.2021.118052](https://doi.org/10.1016/j.neuroimage.2021.118052)

Yanguang Chen. "An analytical process of spatial autocorrelation functions based on Moran’s index." *PLoS ONE*, 16 (2020). [https://doi.org/10.1371/journal.pone.0249589](https://doi.org/10.1371/journal.pone.0249589)

Jesse Whitehead, Amber L Pearson, R. Lawrenson and P. Atatoa-Carr. "How can the spatial equity of health services be defined and measured? A systematic review of spatial equity definitions and methods." *Journal of Health Services Research & Policy*, 24 (2019): 270 \- 278\. [https://doi.org/10.1177/1355819619837292](https://doi.org/10.1177/1355819619837292)

Loïc Caroux. "Presence in video games: A systematic review and meta-analysis of the effects of game design choices.." *Applied ergonomics*, 107 (2022): 103936\. [https://doi.org/10.1016/j.apergo.2022.103936](https://doi.org/10.1016/j.apergo.2022.103936)

J. Amorim, J. de Abreu e Silva and J. Gonçalves. "Equity and Spatial Justice Perspectives in Transportation." *Urban Science* (2025). [https://doi.org/10.3390/urbansci9050163](https://doi.org/10.3390/urbansci9050163)

Izzy Yi Jian, Jiemei Luo and E. Chan. "Spatial justice in public open space planning: Accessibility and inclusivity." *Habitat International*, 97 (2020): 102122\. [https://doi.org/10.1016/j.habitatint.2020.102122](https://doi.org/10.1016/j.habitatint.2020.102122)

Da He, Qian Shi, Jingqian Xue, Peter M. Atkinson and Xiaoping Liu. "Very fine spatial resolution urban land cover mapping using an explicable sub-pixel mapping network based on learnable spatial correlation." *Remote Sensing of Environment* (2023). [https://doi.org/10.1016/j.rse.2023.113884](https://doi.org/10.1016/j.rse.2023.113884)

Gang Xue, Long Ren, Xiaojie Yan, Ziruo Cui, Daqing Gong and Jian Chen. "EXPRESS: Improving the Spatial Fairness of Spatiotemporal Sequence Forecasting Service Systems." *Production and Operations Management* (2025). [https://doi.org/10.1177/10591478251356940](https://doi.org/10.1177/10591478251356940)

\---

# **No, greedy hill-climbing algorithms in procedural content generation do not reliably lead to optimal equilibrium in symmetric maps.**

## **1\. Introduction**

Procedural content generation (PCG) often leverages greedy hill-climbing algorithms for tasks such as map creation and balancing, especially in games requiring symmetric and fair environments. While these algorithms are attractive due to their simplicity and speed, research consistently shows that they tend to converge to local optima rather than global, optimal equilibria—particularly in complex or high-dimensional search spaces typical of game maps (De Araújo et al. 2020, 163 \- 175; Johnson and Jacobson 2002, 359-373; Al-Betar 2016, 153 \- 168; Wu et al. 2008, 396-403). Even when symmetry is enforced by design, the optimization landscape remains rugged, making it difficult for greedy approaches to escape suboptimal solutions without additional mechanisms like random restarts or hybridization with global search methods (De Araújo et al. 2020, 163 \- 175; Grichshenko et al. 2020; Al-Betar 2016, 153 \- 168). Studies comparing hill-climbing with other metaheuristics (e.g., evolutionary algorithms, particle swarm optimization) find that population-based or multi-objective approaches are more effective at achieving balanced and high-quality outcomes in PCG tasks (De Araújo et al. 2020, 163 \- 175; Lianbo et al. 2022, 341-354; Alyaseri et al. 2025, 1-33). Thus, while greedy hill-climbing can produce acceptable results under certain conditions, it does not guarantee optimal equilibrium in symmetric map generation.

**Figure 1:** Consensus that local search rarely guarantees global optima.

## **2\. Methods**

A comprehensive literature search was conducted across over 170 million research papers indexed by Consensus, including sources such as Semantic Scholar and PubMed. The search strategy targeted foundational theories of hill-climbing, algorithmic guarantees, failure modes in PCG, and comparative studies with alternative metaheuristics. In total, 997 papers were identified; after screening and eligibility filtering, 398 were deemed relevant. The final review included the top 50 most pertinent papers.

| Identification | Screening | Eligibility | Included |
| ----- | ----- | ----- | ----- |
| 997 | 598 | 398 | 50 |

**Figure 2:** Flow diagram of paper selection process for this review.

Eight unique search groups were used to ensure broad coverage of both theoretical and applied aspects of greedy hill-climbing in PCG.

## **3\. Results**

### **3.1 Hill-Climbing Performance in PCG**

Greedy hill-climbing is widely used for map generation due to its efficiency but is prone to getting stuck in local optima (De Araújo et al. 2020, 163 \- 175; Al-Betar 2016, 153 \- 168; Johnson and Jacobson 2002, 359-373). For example, steepest-ascent hill climbing with random restarts was found competitive for generating balanced Terra Mystica board game maps but did not guarantee global optimality or unique equilibrium (De Araújo et al. 2020, 163 \- 175).

### **3.2 Symmetry and Equilibrium Challenges**

Even when symmetry constraints are imposed on map representations or fitness functions, the underlying optimization landscape remains multi-modal. This means multiple local optima exist—many of which may not correspond to globally balanced or fair solutions (Lianbo et al. 2022, 341-354; Zafar et al. 2020). Studies show that enforcing symmetry does not resolve the fundamental limitations of greedy local search (De Araújo et al. 2020, 163 \- 175; Lianbo et al. 2022, 341-354).

### **3.3 Comparative Algorithmic Analyses**

Comparisons between greedy hill-climbing and other metaheuristics (e.g., evolutionary algorithms, particle swarm optimization) reveal that population-based methods are more robust at escaping local optima and exploring diverse regions of the solution space (Alyaseri et al. 2025, 1-33; Lianbo et al. 2022, 341-354). Multi-objective evolutionary algorithms explicitly model trade-offs between fairness, playability, and other objectives—yielding a Pareto front rather than a single “optimal” solution (Lianbo et al. 2022, 341-354).

### **3.4 Hybrid Approaches and Enhancements**

Hybridizing greedy hill-climbing with diversity mechanisms (e.g., random restarts, tabu lists) or embedding it within broader metaheuristic frameworks improves performance but still falls short of guaranteeing global equilibrium (Grichshenko et al. 2020; Wang et al. 2009, 763-780; Wu et al. 2008, 396-403). These enhancements help mitigate stagnation but do not fundamentally alter the convergence properties of pure greedy approaches.

#### **Results Timeline**

* **2002**  
  * 1 paper: (Johnson and Jacobson 2002, 359-373)- **2006**  
  * 1 paper: (Tsamardinos et al. 2006, 31-78)- **2009**  
  * 1 paper: (Wang et al. 2009, 763-780)- **2010**  
  * 1 paper: (Wilt et al. 2010, 129-136)- **2016**  
  * 1 paper: (Al-Betar 2016, 153 \- 168)- **2017**  
  * 1 paper: (Shehab et al. 2017, 36-43)- **2019**  
  * 1 paper: (Zafar et al. 2020)- **2020**  
  * 3 papers: (De Araújo et al. 2020, 163 \- 175; Liu et al. 2020, 19 \- 37; Grichshenko et al. 2020)- **2021**  
  * 1 paper: (Arram et al. 2021, 115972-115989)- **2022**  
  * 2 papers: (Garmendia et al. 2022, 18300-18312; Fickert and Hoffmann 2022, 67-115)- **2023**  
  * 3 papers: (Al-Qablan et al. 2023, 59866-59881; Tiwari 2023, 93511-93537; Volz et al. 2023, 110121)- **2024**  
  * 2 papers: (Rodríguez-Esparza et al. 2024, 111784; Chicano et al. 2024)- **2025**  
  * 2 papers: (Alyaseri et al. 2025, 1-33; Yavuz et al. 2025\)**Figure 3:** Timeline showing evolution of research on hill-climbing and PCG balance. Larger markers indicate more citations.

#### **Top Contributors**

| Type | Name | Papers |
| ----- | ----- | ----- |
| Author | M. Al-Betar | (Al-Betar 2016, 153 \- 168; Shehab et al. 2017, 36-43; Al-Betar et al. 2020, 7637 \- 7665\) |
| Author | Alan W. Johnson | (Johnson and Jacobson 2002, 359-373; Johnson and Jacobson 2002, 37-57; Johnson and Jacobson 1996\) |
| Author | S. Jacobson | (Johnson and Jacobson 2002, 359-373; Sullivan and Jacobson 2001, 1288-1293; Johnson and Jacobson 2002, 37-57; Jacobson and Yücesan 2004, 173-190; Jacobson and Yücesan 2004, 387-405) |
| Journal | *IEEE Access* | (Al-Qablan et al. 2023, 59866-59881; Arram et al. 2021, 115972-115989; Tiwari 2023, 93511-93537; Emambocus et al. 2022, 95030-95045) |
| Journal | *Knowl. Based Syst.* | (Rodríguez-Esparza et al. 2024, 111784; Ahmed et al. 2021, 107283\) |
| Journal | *Appl. Soft Comput.* | (Volz et al. 2023, 110121; Zafar et al. 2020\) |

**Figure 4:** Authors & journals that appeared most frequently in the included papers.

## **4\. Discussion**

The reviewed literature demonstrates that while greedy hill-climbing is a practical tool for procedural map generation—especially when computational resources are limited—it is fundamentally limited by its tendency to converge on local rather than global optima (De Araújo et al. 2020, 163 \- 175; Al-Betar 2016, 153 \- 168; Johnson and Jacobson 2002, 359-373). This limitation persists even when symmetry is enforced at the representation level or through constraints; the ruggedness of the fitness landscape means many locally balanced solutions may still be globally suboptimal (Lianbo et al. 2022, 341-354; Zafar et al. 2020).

Population-based metaheuristics (e.g., genetic algorithms) and multi-objective approaches offer superior performance by maintaining diversity and explicitly modeling trade-offs between competing objectives such as fairness and playability (Alyaseri et al. 2025, 1-33; Lianbo et al. 2022, 341-354). Hybrid strategies incorporating restarts or memory structures (e.g., tabu lists) can improve exploration but do not guarantee convergence to an optimal equilibrium without exhaustive search—which is infeasible for large-scale problems (Grichshenko et al. 2020; Wang et al. 2009, 763-780).

Theoretical analyses confirm these empirical findings: only under restrictive conditions (such as known global optimum value or specific move rules) can generalized hill-climbing be proven to converge globally—conditions rarely met in real-world PCG scenarios (Johnson and Jacobson 2002, 359-373; Sullivan and Jacobson 2001, 1288-1293).

### **Claims and Evidence Table**

| Claim | Evidence Strength | Reasoning | Papers |
| ----- | ----- | ----- | ----- |
| Greedy hill-climbing converges to local optima rather than global | Evidence strength: Strong (9/10) | Empirical studies show frequent stagnation at suboptimal solutions; theory supports only local convergence | (De Araújo et al. 2020, 163 \- 175; Al-Betar 2016, 153 \- 168; Johnson and Jacobson 2002, 359-373) |
| Enforcing symmetry does not guarantee global equilibrium | Evidence strength: Strong (8/10) | Symmetry constraints reduce but do not eliminate multi-modality; many locally symmetric solutions persist | (De Araújo et al. 2020, 163 \- 175; Lianbo et al. 2022, 341-354) |
| Population-based/metaheuristic methods outperform pure hill-climbing | Evidence strength: Strong (8/10) | Comparative studies show better exploration/diversity leads to higher-quality/balanced maps | (Alyaseri et al. 2025, 1-33; Lianbo et al. 2022, 341-354) |
| Hybridizing with restarts/memory improves results but lacks guarantees | Evidence strength: Moderate (6/10) | Restarts/tabu lists help escape some local optima but cannot ensure global optimality | (Grichshenko et al. 2020; Wang et al. 2009, 763-780) |
| Theoretical convergence requires restrictive conditions | Evidence strength: Moderate (6/10) | Proofs require knowledge of global optimum value or special move rules | (Johnson and Jacobson 2002, 359-373; Sullivan and Jacobson 2001, 1288-1293) |
| Greedy approaches suffice only for simple/low-dimensional problems | Evidence strength: Moderate (4/10) | In small/simple landscapes local optima may coincide with global ones | (Selman and Gomes 2006\) |

**Figure undefined:** Key claims and support evidence identified in these papers.

## **5\. Conclusion**

Greedy hill-climbing algorithms are useful tools for procedural content generation but do not reliably yield globally optimal equilibria—even on symmetric maps—due to their inherent limitations in complex search spaces. Population-based metaheuristics and multi-objective approaches provide more robust alternatives for achieving balanced outcomes.

### **Research Gaps**

Despite extensive study on algorithmic performance in PCG tasks, there remain gaps regarding formal guarantees of equilibrium quality under realistic constraints and scalability limits.

#### **Research Gaps Matrix**

| Topic/Outcome | Symmetry Constraints Applied | Population-Based Methods Used | Hybrid/Restart Strategies Used | Formal Optimality Guarantees |
| ----- | ----- | ----- | ----- | ----- |
| Map Balance | **7** | **8** | **6** | **2** |
| Equilibrium Quality | **6** | **7** | **5** | **2** |
| Algorithm Scalability | **4** | **6** | **4** | **GAP** |

**Figure undefined:** Matrix showing coverage of key topics versus algorithmic attributes; formal guarantees remain rare.

### **Open Research Questions**

Future work should focus on developing scalable methods with provable guarantees for equilibrium quality under realistic PCG constraints.

| Question | Why |
| ----- | ----- |
| **Can hybrid metaheuristics provide formal guarantees for equilibrium quality in large-scale symmetric map generation?** | Most current methods lack theoretical guarantees; bridging this gap would improve reliability for designers. |
| **How does landscape ruggedness affect the ability of local search methods to find balanced solutions?** | Understanding this relationship could inform better algorithm selection/design based on problem structure. |

**Figure undefined:** Open questions highlight directions for improving theoretical rigor and practical effectiveness.

In summary: Greedy hill-climbing alone is insufficient for reliably achieving optimal equilibria in symmetric procedural map generation; advanced hybrid/metaheuristic strategies offer better prospects but require further research into scalability and formal guarantees.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Luiz Jonatã Pires de Araújo, Alexandr Grichshenko, R. Pinheiro, R. D. Saraiva and Susanna Gimaeva. "Map Generation and Balance in the Terra Mystica Board Game Using Particle Swarm and Local Search." *Advances in Swarm Intelligence*, 12145 (2020): 163 \- 175\. [https://doi.org/10.1007/978-3-030-53956-6\_15](https://doi.org/10.1007/978-3-030-53956-6_15)

Tamara Amjad Al-Qablan, M. H. M. Noor, M. Al-betar and A. Khader. "Improved Binary Gray Wolf Optimizer Based on Adaptive β-Hill Climbing for Feature Selection." *IEEE Access*, 11 (2023): 59866-59881. [https://doi.org/10.1109/access.2023.3285815](https://doi.org/10.1109/access.2023.3285815)

E. Rodríguez-Esparza, Bernardo Morales-Castañeda, Ángel Casas-Ordaz, Diego Oliva, M. Navarro, Arturo Valdivia and Essam H. Houssein. "Handling the balance of operators in evolutionary algorithms through a weighted Hill Climbing approach." *Knowl. Based Syst.*, 294 (2024): 111784\. [https://doi.org/10.1016/j.knosys.2024.111784](https://doi.org/10.1016/j.knosys.2024.111784)

Anas Arram, M. Ayob and A. Sulaiman. "Hybrid Bird Mating Optimizer With Single-Based Algorithms for Combinatorial Optimization Problems." *IEEE Access*, 9 (2021): 115972-115989. [https://doi.org/10.1109/access.2021.3102154](https://doi.org/10.1109/access.2021.3102154)

M. Al-Betar. "β\\documentclass\[12pt\]{minimal} \\usepackage{amsmath} \\usepackage{wasysym} \\usepackage{amsfonts} \\usepackage{amssymb} \\usepackage{amsbsy} \\usepackage{mathrsfs} \\usepackage{upgreek} \\setlength{\\oddsidemargin}{-69pt} \\begin{document}$$\\beta$$\\end{document}-Hill climbing: an exploratory local search." *Neural Computing and Applications*, 28 (2016): 153 \- 168\. [https://doi.org/10.1007/s00521-016-2328-2](https://doi.org/10.1007/s00521-016-2328-2)

I. Tsamardinos, Laura E. Brown and C. Aliferis. "The max-min hill-climbing Bayesian network structure learning algorithm." *Machine Learning*, 65 (2006): 31-78. [https://doi.org/10.1007/s10994-006-6889-7](https://doi.org/10.1007/s10994-006-6889-7)

Anurag Tiwari. "A Hybrid Feature Selection Method Using an Improved Binary Butterfly Optimization Algorithm and Adaptive β–Hill Climbing." *IEEE Access*, 11 (2023): 93511-93537. [https://doi.org/10.1109/access.2023.3274469](https://doi.org/10.1109/access.2023.3274469)

Andoni I. Garmendia, Josu Ceberio and A. Mendiburu. "Neural Improvement Heuristics for Graph Combinatorial Optimization Problems." *IEEE Transactions on Neural Networks and Learning Systems*, 35 (2022): 18300-18312. [https://doi.org/10.1109/tnnls.2023.3314375](https://doi.org/10.1109/tnnls.2023.3314375)

Jialin Liu, Sam Snodgrass, A. Khalifa, S. Risi, Georgios N. Yannakakis and J. Togelius. "Deep learning for procedural content generation." *Neural Computing and Applications*, 33 (2020): 19 \- 37\. [https://doi.org/10.1007/s00521-020-05383-8](https://doi.org/10.1007/s00521-020-05383-8)

Hongfeng Wang, Dingwei Wang and Shengxiang Yang. "A memetic algorithm with adaptive hill climbing strategy for dynamic optimization problems." *Soft Computing*, 13 (2009): 763-780. [https://doi.org/10.1007/s00500-008-0347-3](https://doi.org/10.1007/s00500-008-0347-3)

Vanessa Volz, B. Naujoks, P. Kerschke and Tea Tušar. "Tools for Landscape Analysis of Optimisation Problems in Procedural Content Generation for Games." *Appl. Soft Comput.*, 136 (2023): 110121\. [https://doi.org/10.1016/j.asoc.2023.110121](https://doi.org/10.1016/j.asoc.2023.110121)

Francisco Chicano, Darrell Whitley, Gabriela Ochoa and R. Tin'os. "Generalizing and Unifying Gray-box Combinatorial Optimization Operators." *ArXiv*, abs/2407.06742 (2024). [https://doi.org/10.48550/arxiv.2407.06742](https://doi.org/10.48550/arxiv.2407.06742)

Adeel Zafar, Hasan Mujtaba and M. O. Beg. "Search-based procedural content generation for GVG-LG." *Appl. Soft Comput.*, 86 (2020). [https://doi.org/10.1016/j.asoc.2019.105909](https://doi.org/10.1016/j.asoc.2019.105909)

Alexandr Grichshenko, Luiz Jonatã Pires de Araújo, Susanna Gimaeva and J. A. Brown. "Using Tabu Search Algorithm for Map Generation in the Terra Mystica Tabletop Game." *Proceedings of the 2020 4th International Conference on Intelligent Systems, Metaheuristics & Swarm Intelligence* (2020). [https://doi.org/10.1145/3396474.3396492](https://doi.org/10.1145/3396474.3396492)

Alan W. Johnson and S. Jacobson. "A class of convergent generalized hill climbing algorithms." *Appl. Math. Comput.*, 125 (2002): 359-373. [https://doi.org/10.1016/s0096-3003(00)00137-5](https://doi.org/10.1016/s0096-3003\(00\)00137-5)

Sana Alyaseri, Andy M. Connor and Roopak Sinha. "Exploring Metaheuristic Algorithms for Enhanced Game Map Generation in Procedural Content Generation." *Int. J. Appl. Metaheuristic Comput.*, 16 (2025): 1-33. [https://doi.org/10.4018/ijamc.388932](https://doi.org/10.4018/ijamc.388932)

Mohammad Shehab, A. Khader, M. Al-Betar and L. Abualigah. "Hybridizing cuckoo search algorithm with hill climbing for numerical optimization problems." *2017 8th International Conference on Information Technology (ICIT)* (2017): 36-43. [https://doi.org/10.1109/icitech.2017.8079912](https://doi.org/10.1109/icitech.2017.8079912)

C. Wilt, J. Thayer and Wheeler Ruml. "A Comparison of Greedy Search Algorithms." \*\* (2010): 129-136. [https://doi.org/10.1609/socs.v1i1.18182](https://doi.org/10.1609/socs.v1i1.18182)

Maximilian Fickert and Jörg Hoffmann. "Online Relaxation Refinement for Satisficing Planning: On Partial Delete Relaxation, Complete Hill-Climbing, and Novelty Pruning." *J. Artif. Intell. Res.*, 73 (2022): 67-115. [https://doi.org/10.1613/jair.1.13153](https://doi.org/10.1613/jair.1.13153)

Gürcan Yavuz, Hatem Dumlu and Yacoub Kadar Hassan. "Competitive hybrid Jaya and β-Hill Climbing algorithm for high-cost optimization problems." *Ömer Halisdemir Üniversitesi Mühendislik Bilimleri Dergisi* (2025). [https://doi.org/10.28948/ngumuh.1570577](https://doi.org/10.28948/ngumuh.1570577)

M. Al-Betar, M. Al-Betar, Abdelaziz I. Hammouri, Mohammed A. Awadallah, Iyad Abu Doush and Iyad Abu Doush. "Binary β\\documentclass\[12pt\]{minimal} \\usepackage{amsmath} \\usepackage{wasysym} \\usepackage{amsfonts} \\usepackage{amssymb} \\usepackage{amsbsy} \\usepackage{mathrsfs} \\usepackage{upgreek} \\setlength{\\oddsidemargin}{-69pt} \\begin{document}$$\\beta$$\\end{document}-hill climbing optimizer with S-shape tra." *Journal of Ambient Intelligence and Humanized Computing*, 12 (2020): 7637 \- 7665\. [https://doi.org/10.1007/s12652-020-02484-z](https://doi.org/10.1007/s12652-020-02484-z)

Kelly A. Sullivan and S. Jacobson. "A convergence analysis of generalized hill climbing algorithms." *IEEE Trans. Autom. Control.*, 46 (2001): 1288-1293. [https://doi.org/10.1109/9.940936](https://doi.org/10.1109/9.940936)

Lianbo Ma, Shi Cheng, Mingli Shi and Yi-nan Guo. "Angle-Based Multi-Objective Evolutionary Algorithm Based On Pruning-Power Indicator for Game Map Generation." *IEEE Transactions on Emerging Topics in Computational Intelligence*, 6 (2022): 341-354. [https://doi.org/10.1109/tetci.2021.3067104](https://doi.org/10.1109/tetci.2021.3067104)

Alan W. Johnson and S. Jacobson. "On the convergence of generalized hill climbing algorithms." *Discret. Appl. Math.*, 119 (2002): 37-57. [https://doi.org/10.1016/s0166-218x(01)00264-5](https://doi.org/10.1016/s0166-218x\(01\)00264-5)

Bibi Aamirah Shafaa Emambocus, Muhammed Basheer Jasser and A. Amphawan. "An Optimized Continuous Dragonfly Algorithm Using Hill Climbing Local Search to Tackle the Low Exploitation Problem." *IEEE Access*, 10 (2022): 95030-95045. [https://doi.org/10.1109/access.2022.3204752](https://doi.org/10.1109/access.2022.3204752)

Alan W. Johnson and S. Jacobson. "Generalized hill-climbing algorithms for discrete optimization problems." \*\* (1996).

S. Jacobson and E. Yücesan. "Global Optimization Performance Measures for Generalized Hill Climbing Algorithms." *Journal of Global Optimization*, 29 (2004): 173-190. [https://doi.org/10.1023/b:jogo.0000042111.72036.11](https://doi.org/10.1023/b:jogo.0000042111.72036.11)

B. Selman and C. Gomes. "Hill‐climbing Search." \*\* (2006). [https://doi.org/10.1002/0470018860.s00015](https://doi.org/10.1002/0470018860.s00015)

S. Jacobson and E. Yücesan. "Analyzing the Performance of Generalized Hill Climbing Algorithms." *Journal of Heuristics*, 10 (2004): 387-405. [https://doi.org/10.1023/b:heur.0000034712.48917.a9](https://doi.org/10.1023/b:heur.0000034712.48917.a9)

Shameem Ahmed, K. Ghosh, S. Mirjalili and R. Sarkar. "AIEOU: Automata-based improved equilibrium optimizer with U-shaped transfer function for feature selection." *Knowl. Based Syst.*, 228 (2021): 107283\. [https://doi.org/10.1016/j.knosys.2021.107283](https://doi.org/10.1016/j.knosys.2021.107283)

Jia-Hong Wu, Rajesh Kalyanam and R. Givan. "Stochastic Enforced Hill-Climbing." \*\* (2008): 396-403. [https://doi.org/10.1613/jair.3420](https://doi.org/10.1613/jair.3420)

\---

# **Yes, Tabu Search is effective for escaping local optima in procedural map generation.**

## **1\. Introduction**

Tabu Search (TS) is a metaheuristic optimization technique designed to overcome the limitations of traditional local search algorithms, particularly their tendency to become trapped in local optima. In the context of procedural map generation, TS has been shown to enhance solution quality by enabling the search process to escape local optima and explore a broader solution space. For example, research applying TS to generate maps for the game Terra Mystica demonstrated that TS could produce higher-quality maps than simple hill-climbing by effectively escaping local optima through adaptive memory and tabu list strategies (Grichshenko et al. 2020). Foundational studies on TS further confirm its ability to guide searches beyond local optimality using memory-based strategies and adaptive mechanisms (Subash et al. 2022; Hanafi et al. 2023, 1037-1055; Glover et al. 2018, 361-377; Glover 1990, 74-94; Glover 1989, 190-206). Variants and hybridizations of TS, such as double neighborhood strategies and integration with genetic algorithms, have also been developed to further improve its performance in complex optimization landscapes (Umam et al. 2021, 7459-7467; Glover 1989, 4-32; Glover et al. 1995, 111-134). These findings collectively support the effectiveness of Tabu Search for escaping local optima in procedural map generation and related combinatorial optimization problems.

**Figure 2:** Consensus on Tabu Search's effectiveness in escaping local optima.

## **2\. Methods**

A comprehensive literature search was conducted across over 170 million research papers indexed by Consensus, including sources such as Semantic Scholar and PubMed. The search strategy targeted papers discussing Tabu Search, local optima, and procedural map or content generation, as well as foundational works on TS theory and hybrid metaheuristics. In total, 1007 papers were identified, 651 were screened after de-duplication, 529 met eligibility criteria based on relevance and quality, and the top 50 most relevant papers were included in this review.

| Identification | Screening | Eligibility | Included |
| ----- | ----- | ----- | ----- |
| 1007 | 651 | 529 | 50 |

**Figure 3:** Flow diagram of paper selection process.

Eight unique search groups were used to ensure comprehensive coverage of both theoretical foundations and practical applications of Tabu Search for escaping local optima in procedural map generation.

## **3\. Results**

### **3.1 Application of Tabu Search in Procedural Map Generation**

TS has been directly applied to procedural map generation tasks, such as generating balanced maps for board games like Terra Mystica. These studies found that TS outperformed simple hill-climbing by escaping local optima through adaptive memory structures (tabu lists) and careful tuning of neighborhood size (Grichshenko et al. 2020).

### **3.2 Mechanisms for Escaping Local Optima**

TS employs adaptive memory (tabu lists) to prevent revisiting recently explored solutions, allowing it to accept non-improving moves when necessary (Subash et al. 2022; Hanafi et al. 2023, 1037-1055; Glover et al. 2018, 361-377; Glover 1990, 74-94). Advanced memory mechanisms like exponential extrapolation further enhance TS’s ability to avoid cycling back into previous basins of attraction (Bentsen et al. 2022, 100028; Hanafi et al. 2023, 1037-1055).

### **3.3 Hybrid Approaches and Comparative Performance**

Hybrid algorithms combining TS with genetic algorithms or other metaheuristics leverage TS’s strong local search capabilities to help global search methods avoid premature convergence at local optima (Umam et al. 2021, 7459-7467; Glover 1989, 4-32; Glover et al. 1995, 111-134). Empirical results show that these hybrids often outperform standalone approaches in both convergence speed and solution quality.

### **3.4 Theoretical Foundations and Generalizability**

Foundational works consistently highlight TS’s effectiveness at overcoming local optimality across a wide range of combinatorial optimization problems—attributes that translate well into procedural content generation domains (Glover et al. 2018, 361-377; Glover 1990, 74-94; Glover 1989, 190-206).

#### **Results Timeline**

* **1989**  
  * 2 papers: (Glover 1989, 4-32; Glover 1989, 190-206)- **1990**  
  * 1 paper: (Glover 1990, 74-94)- **1994**  
  * 1 paper: (Glover et al. 1995, 111-134)- **2017**  
  * 1 paper: (Kuo and Chou 2017, 13236-13252)- **2018**  
  * 3 papers: (Balan and Luo 2018, 497-501; Sivaram et al. 2019; Glover et al. 2018, 361-377)- **2020**  
  * 4 papers: (Grichshenko et al. 2020; Silva et al. 2020, 1066-1084; Ghany et al. 2020, 832-839; Cheikh-Graiet et al. 2020, 106217)- **2021**  
  * 3 papers: (Bentsen et al. 2022, 100028; Umam et al. 2021, 7459-7467; Amaral et al. 2022, 219-230)- **2022**  
  * 2 papers: (Subash et al. 2022; Hanafi et al. 2023, 1037-1055)- **2023**  
  * 1 paper: (Han et al. 2023, 1-15)- **2024**  
  * 2 papers: (Niroumandrad et al. 2024, 106751; Szénási and Légrádi 2024, 125192\)**Figure 4:** Chronological distribution of key studies on Tabu Search's role in escaping local optima. Larger markers indicate more citations.

#### **Top Contributors**

| Type | Name | Papers |
| ----- | ----- | ----- |
| Author | F. Glover | (Glover 1989, 4-32; Glover 1990, 74-94; Glover et al. 1995, 111-134; Glover 1989, 190-206; Glover and Taillard 1993, 1-28; Glover 1990, 365-375; Yagiura et al. 2004, 133-151) |
| Author | M. Laguna | (Glover et al. 2018, 361-377; Glover et al. 1995, 111-134; Martí et al. 2007; López-Sánchez et al. 2020, 629-642) |
| Author | Ruizhi Li | (Li et al. 2016, 1775 \- 1785; Li et al. 2016, 1059 \- 1067\) |
| Journal | *INFORMS J. Comput.* | (Glover 1989, 4-32; Glover 1989, 190-206; Battiti and Tecchiolli 1994, 126-140; Skorin-Kapov 1990, 33-45; Yagiura et al. 2004, 133-151) |
| Journal | *Eur. J. Oper. Res.* | (Hanafi et al. 2023, 1037-1055; Silva et al. 2020, 1066-1084) |
| Journal | *J. King Saud Univ. Comput. Inf. Sci.* | (Umam et al. 2021, 7459-7467; Ghany et al. 2020, 832-839; Arık 2021, 7775-7789) |

**Figure 5:** Authors & journals that appeared most frequently in the included papers.

## **4\. Discussion**

The reviewed literature provides strong evidence that Tabu Search is highly effective at escaping local optima due to its use of adaptive memory structures (tabu lists), acceptance of non-improving moves under certain conditions (aspiration criteria), and diversification strategies (Grichshenko et al. 2020; Subash et al. 2022; Hanafi et al. 2023, 1037-1055; Glover et al. 2018, 361-377; Glover 1990, 74-94). These features allow TS to explore new regions of the solution space when traditional methods would stagnate.

In procedural map generation specifically, empirical studies demonstrate that TS can generate higher-quality maps than simpler heuristics by avoiding entrapment in locally optimal but globally suboptimal configurations (Grichshenko et al. 2020). Hybrid approaches further enhance performance by combining global exploration with TS’s robust exploitation capabilities (Umam et al. 2021, 7459-7467; Glover 1989, 4-32).

However, the efficiency of TS depends on careful parameter tuning (e.g., tabu list size) and problem-specific adaptations (Amaral et al. 2022, 219-230). While most evidence is positive regarding its effectiveness for escaping local optima across various domains—including scheduling and layout problems—some challenges remain regarding computational cost for very large or highly complex landscapes.

### **Claims and Evidence Table**

| Claim | Evidence Strength | Reasoning | Papers |
| ----- | ----- | ----- | ----- |
| Tabu Search effectively escapes local optima in procedural map generation | Evidence strength: Strong (9/10) | Direct applications show improved results over hill-climbing; adaptive memory prevents cycling | (Grichshenko et al. 2020; Subash et al. 2022; Hanafi et al. 2023, 1037-1055) |
| Adaptive memory structures are key for avoiding cycles/local traps | Evidence strength: Strong (9/10) | Foundational theory supports this mechanism as central to TS’s success | (Glover et al. 2018, 361-377; Glover 1990, 74-94; Glover 1989, 190-206) |
| Hybridizing TS with other metaheuristics improves convergence | Evidence strength: Strong (8/10) | Hybrids leverage strengths of both global (GA) & local (TS) search | (Umam et al. 2021, 7459-7467; Glover 1989, 4-32; Glover et al. 1995, 111-134) |
| Effectiveness depends on parameter tuning | Evidence strength: Moderate (6/10) | Studies note sensitivity to tabu list size/neighborhood structure | (Amaral et al. 2022, 219-230) |
| Computational cost can be high for large/complex problems | Evidence strength: Moderate (4/10) | Some reports mention increased computation time compared with simpler heuristics | (Amaral et al. 2022, 219-230) |
| Not all variants guarantee escape from all types of complex landscapes | Evidence strength: Weak (3/10) | Certain problem instances may still pose challenges despite advanced memory mechanisms | (Amaral et al. 2022, 219-230) |

**Figure undefined:** Key claims and support evidence identified in these papers.

## **5\. Conclusion**

Tabu Search is a robust metaheuristic that excels at escaping local optima through adaptive memory mechanisms—making it highly effective for procedural map generation tasks where traditional heuristics often fail.

### **Research Gaps**

Despite strong evidence supporting its effectiveness overall, there are gaps regarding large-scale real-time applications in dynamic environments or specific genres outside board games.

#### **Research Gaps Matrix**

| Topic/Outcome | Board Game Maps | Scheduling Problems | Clustering/Data Mining | Vehicle Routing/Path Planning |
| ----- | ----- | ----- | ----- | ----- |
| Escaping Local Optima | **1** | **4** | **2** | **2** |
| Parameter Sensitivity | **1** | **2** | **GAP** | **GAP** |
| Real-Time/Dynamic Generation | **GAP** | **GAP** | **GAP** | **GAP** |

**Figure undefined:** Distribution of research focus across application domains; gaps exist especially for real-time/dynamic map generation.

### **Open Research Questions**

Future work should address scalability issues and explore real-time or dynamic procedural content generation using advanced or hybridized forms of Tabu Search.

| Question | Why |
| ----- | ----- |
| **How can Tabu Search be adapted for real-time procedural map generation?** | Real-time applications require fast adaptation; current research focuses mainly on offline scenarios |
| **What are optimal parameter settings for different genres or scales?** | Parameter sensitivity affects performance; systematic guidelines are lacking |

**Figure undefined:** Open questions highlight directions for future research on scalability and adaptability.

In summary: Tabu Search is an effective tool for escaping local optima in procedural map generation but further research is needed on scalability and real-time adaptation across diverse domains.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Alexandr Grichshenko, Luiz Jonatã Pires de Araújo, Susanna Gimaeva and J. A. Brown. "Using Tabu Search Algorithm for Map Generation in the Terra Mystica Tabletop Game." *Proceedings of the 2020 4th International Conference on Intelligent Systems, Metaheuristics & Swarm Intelligence* (2020). [https://doi.org/10.1145/3396474.3396492](https://doi.org/10.1145/3396474.3396492)

Håkon Bentsen, Arild Hoff and L. M. Hvattum. "Exponential extrapolation memory for tabu search." *EURO J. Comput. Optim.*, 10 (2022): 100028\. [https://doi.org/10.1016/j.ejco.2022.100028](https://doi.org/10.1016/j.ejco.2022.100028)

Dr. N. subash, M. Ramachandran, Vimala Saravanan and Vidhya Prasanth. "An Investigation on Tabu Search Algorithms Optimization." *Electrical and Automation Engineering* (2022). [https://doi.org/10.46632/1/1/3](https://doi.org/10.46632/1/1/3)

Karthika Balan and C. Luo. "Optimal Trajectory Planning for Multiple Waypoint Path Planning using Tabu Search." *2018 9th IEEE Annual Ubiquitous Computing, Electronics & Mobile Communication Conference (UEMCON)* (2018): 497-501. [https://doi.org/10.1109/uemcon.2018.8796810](https://doi.org/10.1109/uemcon.2018.8796810)

S. Hanafi, Yang Wang, F. Glover, Wei-Xing Yang and Rick Hennig. "Tabu search exploiting local optimality in binary optimization." *Eur. J. Oper. Res.*, 308 (2023): 1037-1055. [https://doi.org/10.1016/j.ejor.2023.01.001](https://doi.org/10.1016/j.ejor.2023.01.001)

Shu-Yu Kuo and Yao-Hsin Chou. "Entanglement-Enhanced Quantum-Inspired Tabu Search Algorithm for Function Optimization." *IEEE Access*, 5 (2017): 13236-13252. [https://doi.org/10.1109/access.2017.2723538](https://doi.org/10.1109/access.2017.2723538)

Moch Saiful Umam, M. Mustafid and S. Suryono. "A hybrid genetic algorithm and tabu search for minimizing makespan in flow shop scheduling problem." *J. King Saud Univ. Comput. Inf. Sci.*, 34 (2021): 7459-7467. [https://doi.org/10.1016/j.jksuci.2021.08.025](https://doi.org/10.1016/j.jksuci.2021.08.025)

P. Amaral, Ana Mendes and J. M. Espinosa. "A Tabu Search with a Double Neighborhood Strategy." \*\* (2022): 219-230. [https://doi.org/10.1007/978-3-031-10562-3\_16](https://doi.org/10.1007/978-3-031-10562-3_16)

M. Sivaram, K. Batri, A. Mohammed and V. Porkodi. "Exploiting the Local Optima in Genetic Algorithm using Tabu Search." *Indian Journal of Science and Technology* (2019). [https://doi.org/10.17485/ijst/2019/v12i1/139577](https://doi.org/10.17485/ijst/2019/v12i1/139577)

F. Glover, M. Laguna and R. Martí. "Principles and Strategies of Tabu Search." \*\* (2018): 361-377. [https://doi.org/10.1201/9781351236423-21](https://doi.org/10.1201/9781351236423-21)

F. Glover. "Tabu Search \- Part II." *INFORMS J. Comput.*, 2 (1989): 4-32. [https://doi.org/10.1287/ijoc.2.1.4](https://doi.org/10.1287/ijoc.2.1.4)

F. Glover. "Tabu Search: A Tutorial." *Interfaces*, 20 (1990): 74-94. [https://doi.org/10.1287/inte.20.4.74](https://doi.org/10.1287/inte.20.4.74)

Allyson Silva, Leandro C. Coelho and M. Darvish. "Quadratic assignment problem variants: A survey and an effective parallel memetic iterated tabu search." *Eur. J. Oper. Res.*, 292 (2020): 1066-1084. [https://doi.org/10.1016/j.ejor.2020.11.035](https://doi.org/10.1016/j.ejor.2020.11.035)

F. Glover, J. P. Kelly and M. Laguna. "Genetic algorithms and tabu search: Hybrids for optimization." *Comput. Oper. Res.*, 22 (1995): 111-134. [https://doi.org/10.1016/0305-0548(93)e0023-m](https://doi.org/10.1016/0305-0548\(93\)e0023-m)

Baoting Han, Xiaoyao Zheng, Manping Guan, Liping Sun and Yue Zhang. "Personalized Route Recommendation with Hybrid Tabu Search Algorithm Based on Crowdsensing." *Int. J. Intell. Syst.*, 2023 (2023): 1-15. [https://doi.org/10.1155/2023/3054888](https://doi.org/10.1155/2023/3054888)

F. Glover. "Tabu Search \- Part I." *INFORMS J. Comput.*, 1 (1989): 190-206. [https://doi.org/10.1287/ijoc.1.3.190](https://doi.org/10.1287/ijoc.1.3.190)

Nazgol Niroumandrad, N. Lahrichi and Andrea Lodi. "Learning tabu search algorithms: A scheduling application." *Comput. Oper. Res.*, 170 (2024): 106751\. [https://doi.org/10.1016/j.cor.2024.106751](https://doi.org/10.1016/j.cor.2024.106751)

K. K. A. Ghany, A. Abdelaziz, T. H. Soliman and A. Sewisy. "A hybrid modified step Whale Optimization Algorithm with Tabu Search for data clustering." *J. King Saud Univ. Comput. Inf. Sci.*, 34 (2020): 832-839. [https://doi.org/10.1016/j.jksuci.2020.01.015](https://doi.org/10.1016/j.jksuci.2020.01.015)

Sondes Ben Cheikh-Graiet, M. Dotoli and S. Hammadi. "A Tabu Search based metaheuristic for dynamic carpooling optimization." *Comput. Ind. Eng.*, 140 (2020): 106217\. [https://doi.org/10.1016/j.cie.2019.106217](https://doi.org/10.1016/j.cie.2019.106217)

S. Szénási and Gábor Légrádi. "Machine learning aided metaheuristics: A comprehensive review of hybrid local search methods." *Expert Syst. Appl.*, 258 (2024): 125192\. [https://doi.org/10.1016/j.eswa.2024.125192](https://doi.org/10.1016/j.eswa.2024.125192)

R. Martí, M. Laguna and F. Glover. "Principles of Tabu Search." \*\* (2007). [https://doi.org/10.1201/9781420010749.ch23](https://doi.org/10.1201/9781420010749.ch23)

Ruizhi Li, Shuli Hu, Yiyuan Wang and Minghao Yin. "A local search algorithm with tabu strategy and perturbation mechanism for generalized vertex cover problem." *Neural Computing and Applications*, 28 (2016): 1775 \- 1785\. [https://doi.org/10.1007/s00521-015-2172-9](https://doi.org/10.1007/s00521-015-2172-9)

O. A. Arık. "Fuzzy rule-based acceptance criterion in metaheuristic algorithms." *J. King Saud Univ. Comput. Inf. Sci.*, 34 (2021): 7775-7789. [https://doi.org/10.1016/j.jksuci.2021.09.012](https://doi.org/10.1016/j.jksuci.2021.09.012)

F. Glover and É. Taillard. "A user's guide to tabu search." *Annals of Operations Research*, 41 (1993): 1-28. [https://doi.org/10.1007/bf02078647](https://doi.org/10.1007/bf02078647)

R. Battiti and G. Tecchiolli. "The Reactive Tabu Search." *INFORMS J. Comput.*, 6 (1994): 126-140. [https://doi.org/10.1287/ijoc.6.2.126](https://doi.org/10.1287/ijoc.6.2.126)

F. Glover. "Artificial intelligence, heuristic frameworks and tabu search." *Managerial and Decision Economics*, 11 (1990): 365-375. [https://doi.org/10.1002/mde.4090110512](https://doi.org/10.1002/mde.4090110512)

Ruizhi Li, Shuli Hu, Jian Gao, Yupeng Zhou, Yiyuan Wang and Minghao Yin. "GRASP for connected dominating set problems." *Neural Computing and Applications*, 28 (2016): 1059 \- 1067\. [https://doi.org/10.1007/s00521-016-2429-y](https://doi.org/10.1007/s00521-016-2429-y)

J. Skorin-Kapov. "Tabu Search Applied to the Quadratic Assignment Problem." *INFORMS J. Comput.*, 2 (1990): 33-45. [https://doi.org/10.1287/ijoc.2.1.33](https://doi.org/10.1287/ijoc.2.1.33)

M. Yagiura, T. Ibaraki and F. Glover. "An Ejection Chain Approach for the Generalized Assignment Problem." *INFORMS J. Comput.*, 16 (2004): 133-151. [https://doi.org/10.1287/ijoc.1030.0036](https://doi.org/10.1287/ijoc.1030.0036)

A. D. López-Sánchez, J. Sánchez-Oro and M. Laguna. "A New Scatter Search Design for Multiobjective Combinatorial Optimization with an Application to Facility Location." *INFORMS J. Comput.*, 33 (2020): 629-642. [https://doi.org/10.1287/ijoc.2020.0966](https://doi.org/10.1287/ijoc.2020.0966)

\---

# **Simulated annealing can escape local optima and sometimes achieves better global convergence than greedy hill-climbing for game balance optimization, but its practical advantage depends on the problem landscape and algorithm settings.**

## **1\. Introduction**

Game balance optimization often involves complex, multi-modal search spaces where finding globally optimal solutions is challenging. Two widely used metaheuristics for such problems are simulated annealing (SA) and greedy hill-climbing (HC). SA is designed to probabilistically accept worse solutions, allowing it to escape local optima, while HC only accepts improvements, making it susceptible to getting stuck in suboptimal regions. The literature shows that SA can outperform HC in terms of global convergence, especially in rugged landscapes or when the search space contains many local optima (Kirkpatrick et al. 1983, 671 \- 680; Suman and Kumar 2006, 1143-1160; Mao et al. 2024; Franzin and Stützle 2019, 191-206; Jacobson and Yücesan 2004, 387-405). However, empirical results also indicate that the superiority of SA is not universal; its performance depends on factors such as cooling schedules, problem size, and the specific characteristics of the optimization landscape (Gulta and Chen 2024, 1-6; Odeyemi and Zhang 2025; Yafrani and Ahiod 2018, 231-244; Burke and Bykov 2017, 70-78). Hybrid approaches and algorithmic enhancements further blur the distinction between these methods (Lin 2013, 1064-1073; Bisht 2004, 395-405; Jeong and Lee 1996, 523-532). Overall, while SA offers theoretical guarantees for global convergence under certain conditions, practical outcomes vary based on implementation details and problem structure.

**Figure 1:** Consensus on whether simulated annealing outperforms greedy hill-climbing for game balance optimization.

## **2\. Methods**

A comprehensive search was conducted across over 170 million research papers in Consensus, including sources like Semantic Scholar and PubMed. The process identified 1,074 potentially relevant papers, screened 624 after de-duplication, filtered 408 as eligible based on relevance to simulated annealing and hill-climbing in game balance or combinatorial optimization contexts, and included the top 50 most relevant papers for this review.

| Identification | Screening | Eligibility | Included |
| ----- | ----- | ----- | ----- |
| 1074 | 624 | 408 | 50 |

**Figure 2:** Flow diagram of paper selection from identification to inclusion.

Eight unique search strategies were used to capture foundational theory, empirical comparisons, hybrid algorithms, limitations, and applications in game balancing.

## **3\. Results**

### **3.1 Theoretical Convergence Properties**

Simulated annealing has strong theoretical backing for global convergence: with a sufficiently slow cooling schedule (e.g., logarithmic), it converges in probability to a global optimum (Kirkpatrick et al. 1983, 671 \- 680; Hajek 1988, 311-329; Yao and Li 2008, 329-338). In contrast, greedy hill-climbing typically converges only to local optima unless combined with restarts or diversity mechanisms (Jacobson and Yücesan 2004, 387-405; Johnson and Jacobson 2002, 359-373). Generalized frameworks show that both methods can be analyzed under similar conditions but differ fundamentally in their ability to escape local traps (Jacobson and Yücesan 2004, 387-405).

### **3.2 Empirical Performance Comparisons**

Empirical studies demonstrate that SA often outperforms HC on multi-modal or rugged landscapes by avoiding premature convergence (Suman and Kumar 2006, 1143-1160; Mao et al. 2024; Sun et al. 2021). For example, in discrete choice experiment design—a setting analogous to game balancing—SA produced higher-quality solutions than coordinate-exchange (a form of HC) by escaping local optima (Mao et al. 2024). However, some benchmarks reveal that randomized HC with restarts or hybridization can match or exceed SA's performance depending on landscape structure and computational budget (Gulta and Chen 2024, 1-6; Odeyemi and Zhang 2025).

### **3.3 Hybrid and Enhanced Algorithms**

Hybrid algorithms combining elements of SA and HC (or other metaheuristics) frequently achieve superior results compared to either method alone (Lin 2013, 1064-1073; Bisht 2004, 395-405; Jeong and Lee 1996, 523-532). Multi-start strategies—where multiple runs of HC are combined with SA's probabilistic acceptance—can significantly improve solution quality and robustness (Lin 2013, 1064-1073).

### **3.4 Limitations and Landscape Dependence**

The advantage of SA over HC is not universal. On smooth or unimodal landscapes—or when computational resources are limited—HC may converge faster or as effectively as SA (Gulta and Chen 2024, 1-6; Odeyemi and Zhang 2025). Moreover, poorly tuned cooling schedules can cause SA to behave similarly to random search or even underperform compared to advanced variants of HC (Gulta and Chen 2024, 1-6).

#### **Results Timeline**

* **1983**  
  * 1 paper: (Kirkpatrick et al. 1983, 671 \- 680)- **2004**  
  * 1 paper: (Jacobson and Yücesan 2004, 387-405)- **2006**  
  * 2 papers: (Suman and Kumar 2006, 1143-1160; Henderson et al. 2006, 287-319)- **2012**  
  * 1 paper: (Chen et al. 2012, 1-7)- **2013**  
  * 1 paper: (Lin 2013, 1064-1073)- **2017**  
  * 1 paper: (Burke and Bykov 2017, 70-78)- **2018**  
  * 3 papers: (Han et al. 2019, 44391-44403; Yafrani and Ahiod 2018, 231-244; Assad and Deep 2018, 246-266)- **2019**  
  * 1 paper: (Franzin and Stützle 2019, 191-206)- **2021**  
  * 1 paper: (Nawaz et al. 2021, 1 \- 22)- **2022**  
  * 1 paper: (Hashim et al. 2022, 84-110)- **2023**  
  * 2 papers: (Antognini et al. 2023, 1323 \- 1337; Belkourchia et al. 2023, 438 \- 475)- **2024**  
  * 5 papers: (Gulta and Chen 2024, 1-6; Onizawa and Hanyu 2024; Mao et al. 2024; Zandvakili et al. 2024, 11851 \- 11886; Rodríguez-Esparza et al. 2024, 111784\)**Figure 3:** Timeline showing publication trends comparing simulated annealing and hill-climbing for optimization tasks. Larger markers indicate more citations.

#### **Top Contributors**

| Type | Name | Papers |
| ----- | ----- | ----- |
| Author | S. Jacobson | (Jacobson and Yücesan 2004, 387-405; Orosz and Jacobson 2002, 165-182; Johnson and Jacobson 2002, 359-373) |
| Author | Alan W. Johnson | (Henderson et al. 2006, 287-319; Johnson and Jacobson 2002, 359-373) |
| Author | E. Burke | (Burke and Bykov 2017, 70-78) |
| Journal | *Comput. Oper. Res.* | (Franzin and Stützle 2019, 191-206; Ahmed and Alkhamis 2002, 387-402) |
| Journal | *Inf. Sci.* | (Yafrani and Ahiod 2018, 231-244; Assad and Deep 2018, 246-266) |
| Journal | *Engineering Optimization* | (Suppapitnarm et al. 2000, 59 \- 85; Dowsland 1991, 91-107) |

**Figure 4:** Authors & journals that appeared most frequently in the included papers.

## **4\. Discussion**

The literature consistently highlights simulated annealing's strength in escaping local optima due to its probabilistic acceptance mechanism—a key advantage over greedy hill-climbing in complex search spaces typical of game balance problems (Kirkpatrick et al. 1983, 671 \- 680; Suman and Kumar 2006, 1143-1160; Mao et al. 2024). Theoretical work confirms that with appropriate cooling schedules, SA can guarantee global convergence given enough time (Hajek 1988, 311-329; Yao and Li 2008, 329-338), whereas HC generally cannot unless augmented with restarts or diversity mechanisms (Jacobson and Yücesan 2004, 387-405).

However, practical superiority is context-dependent: empirical studies show that on certain landscapes or with limited computational budgets, well-tuned HC (especially with restarts) can perform comparably or even better than standard implementations of SA (Gulta and Chen 2024, 1-6; Odeyemi and Zhang 2025). Hybrid approaches leveraging both methods' strengths are increasingly popular for robust performance across diverse problem types (Lin 2013, 1064-1073; Bisht 2004, 395-405).

The main limitation is that neither method universally dominates; their relative effectiveness hinges on problem structure (e.g., modality), parameter tuning (e.g., cooling schedule), and available computational resources.

### **Claims and Evidence Table**

| Claim | Evidence Strength | Reasoning | Papers |
| ----- | ----- | ----- | ----- |
| Simulated annealing can escape local optima better than greedy HC | Evidence strength: Strong (9/10) | Probabilistic acceptance allows exploration beyond local minima; proven theoretically & empirically | (Kirkpatrick et al. 1983, 671 \- 680; Suman and Kumar 2006, 1143-1160; Mao et al. 2024; Franzin and Stützle 2019, 191-206) |
| Simulated annealing guarantees global convergence under slow cooling | Evidence strength: Strong (10/10) | Mathematical proofs show asymptotic convergence if cooling is sufficiently slow | (Hajek 1988, 311-329; Yao and Li 2008, 329-338) |
| Greedy hill-climbing converges faster but risks premature trapping | Evidence strength: Strong (8/10) | Only accepts improvements; fast but easily stuck without restarts/diversity | (Jacobson and Yücesan 2004, 387-405; Johnson and Jacobson 2002, 359-373) |
| Hybrid/multi-start approaches outperform single-method strategies | Evidence strength: Moderate (7/10) | Combining strengths improves robustness/solution quality | (Lin 2013, 1064-1073; Bisht 2004, 395-405) |
| On some landscapes/benchmarks randomized HC matches/exceeds SA | Evidence strength: Moderate (5/10) | Empirical studies show landscape dependence; no universal dominance | (Gulta and Chen 2024, 1-6; Odeyemi and Zhang 2025\) |
| Poorly tuned SA may perform no better than random search | Evidence strength: Moderate (4/10) | Fast cooling/poor parameters degrade performance | (Gulta and Chen 2024, 1-6) |

**Figure 5:** Key claims and support evidence identified in these papers.

## **5\. Conclusion**

Simulated annealing offers theoretical advantages over greedy hill-climbing for game balance optimization by enabling escape from local optima and guaranteeing global convergence under ideal conditions. However, practical superiority depends on problem complexity and algorithm configuration; hybrid approaches often yield the best results.

### **Research Gaps**

Despite extensive study of both algorithms individually and in combination, there remain gaps regarding their comparative performance across diverse real-world game balancing scenarios—especially large-scale or highly dynamic games.

#### **Research Gaps Matrix**

| Topic/Outcome | Theoretical Analysis | Empirical Benchmarking | Game Balance Applications | Hybrid Algorithm Studies |
| ----- | ----- | ----- | ----- | ----- |
| Global Convergence Proofs | **6** | **2** | **GAP** | **GAP** |
| Local Optima Escape | **5** | **4** | **2** | **2** |
| Practical Game Balance Outcomes | **GAP** | **2** | **6** | **2** |
| Large-Scale/Dynamic Games | **GAP** | **1** | **1** | **GAP** |

**Figure 6:** Matrix showing coverage of topics versus study attributes; gaps highlight areas needing further research.

### **Open Research Questions**

Future research should focus on systematic benchmarking across a wider range of real-world game balancing problems—including large-scale multiplayer games—and explore adaptive/hybrid metaheuristics tailored for dynamic environments.

| Question | Why |
| ----- | ----- |
| **How do simulated annealing and greedy hill-climbing compare for large-scale dynamic game balancing?** | Most studies focus on static/small-scale problems; real-world games present new challenges |
| **What hybrid strategies best combine simulated annealing’s exploration with hill-climbing’s exploitation?** | Hybrids may offer superior robustness/performance but optimal designs remain underexplored |

**Figure 7:** Open questions highlight future directions for comparative algorithm research.

In summary: Simulated annealing provides important advantages over greedy hill-climbing for escaping local optima in complex game balance optimization—but its practical benefit depends strongly on problem specifics and implementation choices; hybrid approaches are promising avenues for future work.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

E. Burke and Yuri Bykov. "The late acceptance Hill-Climbing heuristic." *Eur. J. Oper. Res.*, 258 (2017): 70-78. [https://doi.org/10.1016/j.ejor.2016.07.012](https://doi.org/10.1016/j.ejor.2016.07.012)

M. Nawaz, Philippe Fournier-Viger, Unil Yun, Youxi Wu and Wei Song. "Mining High Utility Itemsets with Hill Climbing and Simulated Annealing." *ACM Transactions on Management Information System (TMIS)*, 13 (2021): 1 \- 22\. [https://doi.org/10.1145/3462636](https://doi.org/10.1145/3462636)

Dawit Gulta and Stephen Chen. "Improvements to Dual Annealing in SciPy." *2024 IEEE Latin American Conference on Computational Intelligence (LA-CCI)* (2024): 1-6. [https://doi.org/10.1109/la-cci62337.2024.10814799](https://doi.org/10.1109/la-cci62337.2024.10814799)

Fatma A. Hashim, Essam H. Houssein, Kashif Hussain, M. Mabrouk and W. Al-Atabany. "Honey Badger Algorithm: New metaheuristic algorithm for solving optimization problems." *Math. Comput. Simul.*, 192 (2022): 84-110. [https://doi.org/10.1016/j.matcom.2021.08.013](https://doi.org/10.1016/j.matcom.2021.08.013)

Stephen Y. Chen, Carlos Xudiera and James Montgomery. "Simulated annealing with thresheld convergence." *2012 IEEE Congress on Evolutionary Computation* (2012): 1-7. [https://doi.org/10.1109/cec.2012.6256591](https://doi.org/10.1109/cec.2012.6256591)

Xiaoxia Han, Yingchao Dong, Lin Yue and Quanxi Xu. "State Transition Simulated Annealing Algorithm for Discrete-Continuous Optimization Problems." *IEEE Access*, 7 (2019): 44391-44403. [https://doi.org/10.1109/access.2019.2908961](https://doi.org/10.1109/access.2019.2908961)

Mohamed El Yafrani and B. Ahiod. "Efficiently solving the Traveling Thief Problem using hill climbing and simulated annealing." *Inf. Sci.*, 432 (2018): 231-244. [https://doi.org/10.1016/j.ins.2017.12.011](https://doi.org/10.1016/j.ins.2017.12.011)

A. Franzin and T. Stützle. "Revisiting simulated annealing: A component-based analysis." *Comput. Oper. Res.*, 104 (2019): 191-206. [https://doi.org/10.1016/j.cor.2018.12.015](https://doi.org/10.1016/j.cor.2018.12.015)

Shih-Wei Lin. "Solving the team orienteering problem using effective multi-start simulated annealing." *Appl. Soft Comput.*, 13 (2013): 1064-1073. [https://doi.org/10.1016/j.asoc.2012.09.022](https://doi.org/10.1016/j.asoc.2012.09.022)

N. Onizawa and Takahiro Hanyu. "Enhanced convergence in p-bit based simulated annealing with partial deactivation for large-scale combinatorial optimization problems." *Scientific Reports*, 14 (2024). [https://doi.org/10.1038/s41598-024-51639-x](https://doi.org/10.1038/s41598-024-51639-x)

B. Suman and P. Kumar. "A survey of simulated annealing as a tool for single and multiobjective optimization." *Journal of the Operational Research Society*, 57 (2006): 1143-1160. [https://doi.org/10.1057/palgrave.jors.2602068](https://doi.org/10.1057/palgrave.jors.2602068)

S. Kirkpatrick, C. D. Gelatt and Mario P. Vecchi. "Optimization by Simulated Annealing." *Science*, 220 (1983): 671 \- 680\. [https://doi.org/10.1126/science.220.4598.671](https://doi.org/10.1126/science.220.4598.671)

Assif Assad and Kusum Deep. "A Hybrid Harmony search and Simulated Annealing algorithm for continuous optimization." *Inf. Sci.*, 450 (2018): 246-266. [https://doi.org/10.1016/j.ins.2018.03.042](https://doi.org/10.1016/j.ins.2018.03.042)

Yicheng Mao, R. Kessels and Tom C. van der Zanden. "Constructing Bayesian optimal designs for discrete choice experiments by simulated annealing." *Journal of Choice Modelling* (2024). [https://doi.org/10.1016/j.jocm.2025.100551](https://doi.org/10.1016/j.jocm.2025.100551)

Alessandro Baldi Antognini, M. Novelli and Maroussa Zagoraiou. "Simulated annealing for balancing covariates." *Statistics in Medicine*, 42 (2023): 1323 \- 1337\. [https://doi.org/10.1002/sim.9672](https://doi.org/10.1002/sim.9672)

Yassin Belkourchia, M. Z. Es-sadek and L. Azrar. "New Hybrid Perturbed Projected Gradient and Simulated Annealing Algorithms for Global Optimization." *Journal of Optimization Theory and Applications*, 197 (2023): 438 \- 475\. [https://doi.org/10.1007/s10957-023-02210-7](https://doi.org/10.1007/s10957-023-02210-7)

S. Jacobson and E. Yücesan. "Analyzing the Performance of Generalized Hill Climbing Algorithms." *Journal of Heuristics*, 10 (2004): 387-405. [https://doi.org/10.1023/b:heur.0000034712.48917.a9](https://doi.org/10.1023/b:heur.0000034712.48917.a9)

A. Zandvakili, N. Mansouri and M. Javidi. "SA-SGA: Simulated Annealing Optimization and Stochastic Gradient Ascent Reinforcement Learning for Feature Selection." *Arabian Journal for Science and Engineering*, 50 (2024): 11851 \- 11886\. [https://doi.org/10.1007/s13369-024-09587-1](https://doi.org/10.1007/s13369-024-09587-1)

E. Rodríguez-Esparza, Bernardo Morales-Castañeda, Ángel Casas-Ordaz, Diego Oliva, M. Navarro, Arturo Valdivia and Essam H. Houssein. "Handling the balance of operators in evolutionary algorithms through a weighted Hill Climbing approach." *Knowl. Based Syst.*, 294 (2024): 111784\. [https://doi.org/10.1016/j.knosys.2024.111784](https://doi.org/10.1016/j.knosys.2024.111784)

Darrall Henderson, S. Jacobson and Alan W. Johnson. "The Theory and Practice of Simulated Annealing." \*\* (2006): 287-319. [https://doi.org/10.1007/0-306-48056-5\_10](https://doi.org/10.1007/0-306-48056-5_10)

J. E. Orosz and S. Jacobson. "Analysis of Static Simulated Annealing Algorithms." *Journal of Optimization Theory and Applications*, 115 (2002): 165-182. [https://doi.org/10.1023/a:1019633214895](https://doi.org/10.1023/a:1019633214895)

B. Hajek. "Cooling Schedules for Optimal Annealing." *Math. Oper. Res.*, 13 (1988): 311-329. [https://doi.org/10.1287/moor.13.2.311](https://doi.org/10.1287/moor.13.2.311)

Mohamed A. Ahmed and Talal M. Alkhamis. "Simulation-based optimization using simulated annealing with ranking and selection." *Comput. Oper. Res.*, 29 (2002): 387-402. [https://doi.org/10.1016/s0305-0548(00)00073-3](https://doi.org/10.1016/s0305-0548\(00\)00073-3)

I. Jeong and Jujang Lee. "Adaptive simulated annealing genetic algorithm for system identification." *Engineering Applications of Artificial Intelligence*, 9 (1996): 523-532. [https://doi.org/10.1016/0952-1976(96)00049-8](https://doi.org/10.1016/0952-1976\(96\)00049-8)

Alan W. Johnson and S. Jacobson. "A class of convergent generalized hill climbing algorithms." *Appl. Math. Comput.*, 125 (2002): 359-373. [https://doi.org/10.1016/s0096-3003(00)00137-5](https://doi.org/10.1016/s0096-3003\(00\)00137-5)

Libo Sun, James Browning and Roberto Perera. "Shedding some light on Light Up with Artificial Intelligence." *ArXiv*, abs/2107.10429 (2021).

Sanjay Bisht. "Hybrid Genetic-simulated Annealing Algorithm for Optimal Weapon Allocation in Multilayer Defence Scenario." *Defence Science Journal*, 54 (2004): 395-405. [https://doi.org/10.14429/dsj.54.2054](https://doi.org/10.14429/dsj.54.2054)

A. Suppapitnarm, K. Seffen, G. Parks and P. Clarkson. "A SIMULATED ANNEALING ALGORITHM FOR MULTIOBJECTIVE OPTIMIZATION." *Engineering Optimization*, 33 (2000): 59 \- 85\. [https://doi.org/10.1080/03052150008940911](https://doi.org/10.1080/03052150008940911)

X. Yao and Guojie Li. "General simulated annealing." *Journal of Computer Science and Technology*, 6 (2008): 329-338. [https://doi.org/10.1007/bf02948392](https://doi.org/10.1007/bf02948392)

Jethro Odeyemi and W. Zhang. "Benchmarking Randomized Optimization Algorithms on Binary, Permutation, and Combinatorial Problem Landscapes." *ArXiv*, abs/2501.17170 (2025). [https://doi.org/10.48550/arxiv.2501.17170](https://doi.org/10.48550/arxiv.2501.17170)

K. Dowsland. "HILL-CLIMBING, SIMULATED ANNEALING AND THE STEINER PROBLEM IN GRAPHS." *Engineering Optimization*, 17 (1991): 91-107. [https://doi.org/10.1080/03052159108941063](https://doi.org/10.1080/03052159108941063)

\---

# **Yes, Bayesian optimization can be suitable for high-dimensional combinatorial search in board game map design, especially when using modern BO variants tailored for discrete and high-dimensional spaces.**

## **1\. Introduction**

Bayesian optimization (BO) is a sample-efficient method for optimizing expensive black-box functions and has been widely adopted in engineering, machine learning, and scientific domains. The challenge of scaling BO to high-dimensional and combinatorial spaces—such as those encountered in board game map design—has spurred the development of specialized algorithms. Recent advances demonstrate that BO can remain effective in such settings by leveraging techniques like local modeling, variable selection, embeddings, and meta-algorithms designed for mixed or discrete variables (Gui et al. 2024, 121056; Li and Zhan 2025, 247-252; Papenmeier et al. 2025; Ngo et al. 2025; Papenmeier et al. 2023; Deshwal et al. 2023, 7021-7039; Daulton et al. 2021; Oh et al. 2019, 2910-2920). While traditional procedural content generation (PCG) in games often relies on evolutionary or swarm-based methods (De Araújo et al. 2020, 163 \- 175), there is growing evidence that BO—when adapted appropriately—can outperform these alternatives in terms of sample efficiency and solution quality (Celemin 2024, 1-4; Duque et al. 2020, 503-510). Thus, Bayesian optimization is not only suitable but also promising for high-dimensional combinatorial search problems in board game map design.

**Figure 1:** Consensus on the suitability of Bayesian optimization for high-dimensional combinatorial search.

## **2\. Methods**

A comprehensive literature search was conducted across over 170 million research papers indexed by Consensus, including sources such as Semantic Scholar and PubMed. The search strategy involved multiple targeted queries focusing on Bayesian optimization in high-dimensional, combinatorial, and game design contexts. In total, 1124 papers were identified, 560 were screened after de-duplication, 414 met relevance criteria, and the top 50 most relevant papers were included in this review.

| Identification | Screening | Eligibility | Included |
| ----- | ----- | ----- | ----- |
| 1124 | 560 | 414 | 50 |

**Figure 2:** Flow diagram of the literature search and selection process.

Eight unique search groups were used to ensure coverage of foundational theory, algorithmic advances, applications to games and PCG, comparisons with alternative methods, and recent breakthroughs.

## **3\. Results**

### **3.1 Advances in High-Dimensional Bayesian Optimization**

Recent work has produced several strategies to overcome the curse of dimensionality in BO:

* Local/global model switching (Gui et al. 2024, 121056; Li and Zhan 2025, 247-252)- Adaptive dropout or coordinate-wise improvement (Huang and Zhan 2025; Zhan 2024)- Subspace selection and variable selection (e.g., Lasso-based or group testing) (Hoang et al. 2025; Hellsten et al. 2023; Hellsten et al. 2025)- Embedding-based approaches (random projections, kernel PCA) (Lu and Zhu 2024; Antonov et al. 2022, 118-131; Moriconi et al. 2019, 1925 \- 1943; Priem et al. 2024)- Meta-algorithms that partition the space or use hierarchical/ensemble models (Ngo et al. 2025; Ngo et al. 2024; Ngo et al. 2025, 19659-19667)These methods consistently outperform standard BO and even some evolutionary baselines on synthetic and real-world benchmarks with hundreds of variables.

### **3.2 Combinatorial & Mixed Variable Spaces**

Specialized algorithms such as COMBO (graph Cartesian product), Bounce (nested embeddings), MOCA-HESP (hyper-ellipsoid partitioning), and dictionary-based embeddings have been developed specifically for combinatorial or mixed discrete-continuous spaces (Ngo et al. 2025; Papenmeier et al. 2023; Deshwal et al. 2023, 7021-7039; Oh et al. 2019, 2910-2920). These approaches enable efficient surrogate modeling and acquisition function optimization even when variables are categorical or ordinal.

### **3.3 Applications to Game Design & Procedural Content Generation**

While most PCG research uses evolutionary algorithms or swarm intelligence for map/level generation (De Araújo et al. 2020, 163 \- 175), there are emerging examples where BO is applied to optimize agent behavior or level difficulty efficiently (Celemin 2024, 1-4; Duque et al. 2020, 503-510). These studies show that BO can achieve better coverage or target properties with fewer evaluations compared to traditional methods.

### **3.4 Comparative Performance & Limitations**

Empirical comparisons indicate that modern high-dimensional BO variants can match or exceed the performance of evolutionary algorithms under limited evaluation budgets—a common constraint in simulation-heavy domains like board game map design (Santoni et al. 2023, 1 \- 33). However, challenges remain if the problem lacks exploitable structure (e.g., no low-dimensional subspaces), though new methods are increasingly robust to such cases (Hvarfner et al. 2024; Xu and Zhe 2024).

#### **Results Timeline**

* **Feb 2019**  
  * 1 paper: (Moriconi et al. 2019, 1925 \- 1943)- **Nov 2019**  
  * 1 paper: (Tran et al. 2019)- **Sep 2021**  
  * 1 paper: (Daulton et al. 2021)- **Apr 2022**  
  * 1 paper: (Antonov et al. 2022, 118-131)- **Mar 2023**  
  * 1 paper: (Deshwal et al. 2023, 7021-7039)- **Apr 2023**  
  * 1 paper: (Bian et al. 2023, 110630)- **Jul 2023**  
  * 1 paper: (Papenmeier et al. 2023)- **Apr 2024**  
  * 1 paper: (Zhan 2024)- **May 2024**  
  * 1 paper: (Gui et al. 2024, 121056)- **Jun 2024**  
  * 2 papers: (Koide and Okoso 2024, 90862-90872; Rashidi et al. 2024, 3502-3510)- **Aug 2024**  
  * 2 papers: (Celemin 2024, 1-4; Lu and Zhu 2024)- **Feb 2025**  
  * 1 paper: (Papenmeier et al. 2025)- **Apr 2025**  
  * 2 papers: (Huang and Zhan 2025; Hoang et al. 2025)- **Aug 2025**  
  * 4 papers: (Li and Zhan 2025, 247-252; Ngo et al. 2025; Jin and Zhan 2025, 80-85; Miao and Zhan 2025, 112-117)**Figure 3:** Timeline of key publications advancing high-dimensional Bayesian optimization. Larger markers indicate more citations.

#### **Top Contributors**

| Type | Name | Papers |
| ----- | ----- | ----- |
| Author | Dawei Zhan | (Gui et al. 2024, 121056; Li and Zhan 2025, 247-252; Huang and Zhan 2025; Zhan 2024; Jin and Zhan 2025, 80-85; Hoang et al. 2025; Miao and Zhan 2025, 112-117; Liu and Zhan 2025, 241-246) |
| Author | Lam Ngo | (Ngo et al. 2025; Ngo et al. 2024; Ngo et al. 2025, 19659-19667) |
| Author | Leonard Papenmeier | (Papenmeier et al. 2025; Papenmeier et al. 2023; Hellsten et al. 2023; Hellsten et al. 2025\) |
| Journal | *ArXiv* | (Papenmeier et al. 2025; Huang and Zhan 2025; Zhan 2024; Ngo et al. 2025; Papenmeier et al. 2023; Hoang et al. 2025; Lu and Zhu 2024; Antonov et al. 2022, 118-131; Deshwal et al. 2023, 7021-7039; Eriksson and Jankowiak 2021; Hellsten et al. 2023; Wan et al. 2021, 10663-10674; Hvarfner et al. 2024; Ngo et al. 2024; Xu and Zhe 2024; Hellsten et al. 2025\) |
| Journal | *IEEE Access* | (Koide and Okoso 2024, 90862-90872; Amini et al. 2025, 1581-1593) |
| Journal | *2025 7th International Conference on Data-driven Optimization of Complex Systems (DOCS)* | (Li and Zhan 2025, 247-252; Jin and Zhan 2025, 80-85; Miao and Zhan 2025, 112-117) |

**Figure 4:** Authors & journals that appeared most frequently in the included papers.

## **4\. Discussion**

The literature demonstrates that Bayesian optimization is increasingly capable of handling high-dimensional combinatorial problems through innovations such as local modeling strategies (Gui et al. 2024, 121056; Li and Zhan 2025, 247-252), embedding techniques (Lu and Zhu 2024; Deshwal et al. 2023, 7021-7039), variable/subspace selection (Hoang et al. 2025; Hellsten et al. 2023), and meta-algorithms like MOCA-HESP (Ngo et al. 2025)or Bounce (Papenmeier et al. 2023). These advances address both scalability and sample efficiency—critical factors when evaluating candidate maps is computationally expensive.

Although direct applications to board game map design are rare compared to evolutionary approaches (De Araújo et al. 2020, 163 \- 175), analogous successes in chip layout design (Shen et al. 2021, 1-3), neural architecture search (Song et al. 2022), materials discovery (Papenmeier et al. 2023), and automated game testing/map coverage tasks suggest strong transferability to PCG contexts. Notably:

* COMBO leverages graph structures for efficient GP modeling over large discrete spaces with variable selection built-in—ideal for complex map parameterizations (Oh et al. 2019, 2910-2920).  
* Dictionary-based embeddings allow standard GPs to operate over binary/categorical structures typical of map layouts (Deshwal et al. 2023, 7021-7039).  
* Multi-objective extensions like MORBO enable simultaneous optimization of multiple gameplay criteria (balance/fun/variety) at scale (Daulton et al. 2021).

However:

* If the objective function lacks any exploitable structure (e.g., all variables equally important), performance may degrade unless advanced randomization/embedding strategies are used (Hvarfner et al. 2024).  
* Most empirical studies focus on synthetic benchmarks; more real-world case studies in PCG/map design would strengthen confidence.

### **Claims and Evidence Table**

| Claim | Evidence Strength | Reasoning | Papers |
| ----- | ----- | ----- | ----- |
| Modern BO variants are effective for high-dimensional combinatorial spaces | Evidence strength: Strong (9/10) | Multiple new algorithms show state-of-the-art results on synthetic/real benchmarks up to hundreds of variables | (Gui et al. 2024, 121056; Li and Zhan 2025, 247-252; Ngo et al. 2025; Papenmeier et al. 2023; Deshwal et al. 2023, 7021-7039; Oh et al. 2019, 2910-2920) |
| Embedding/subspace methods improve scalability | Evidence strength: Strong (8/10) | Embeddings reduce dimensionality; subspace selection focuses computation where it matters | (Lu and Zhu 2024; Antonov et al. 2022, 118-131; Moriconi et al. 2019, 1925 \- 1943; Priem et al. 2024\) |
| BO outperforms evolutionary/sampling baselines under tight budgets | Evidence strength: Moderate (7/10) | Empirical comparisons show higher sample efficiency when evaluations are expensive | (Celemin 2024, 1-4; Santoni et al. 2023, 1 \- 33\) |
| Direct application to board game map design is underexplored | Evidence strength: Moderate (5/10) | Most PCG/game studies use evolutionary/swarm methods; few use advanced BO | (De Araújo et al. 2020, 163 \- 175\) |
| Performance may degrade without structure/exploitable subspaces | Evidence strength: Moderate (4/10) | Some methods rely on axis-aligned subspaces; randomization helps but not always optimal | (Hvarfner et al. 2024\) |
| Standard vanilla BO struggles without adaptation | Evidence strength: Weak (3/10) | Without lengthscale tuning/embeddings/local models performance drops sharply | (Xu and Zhe 2024\) |

**Figure undefined:** Key claims and support evidence identified in these papers.

## **5\. Conclusion**

Bayesian optimization—especially with recent algorithmic advances—is well-suited for high-dimensional combinatorial search problems like board game map design when adapted appropriately. While direct applications remain rare compared to evolutionary approaches in PCG literature so far (De Araújo et al. 2020, 163 \- 175), strong evidence from related domains supports its suitability.

### **Research Gaps**

Despite rapid progress:

* Few studies directly apply advanced BO variants to procedural content generation or board game map design.  
* Most benchmarks remain synthetic; real-world case studies are needed.  
* Multi-objective/multi-modal aspects specific to games require further exploration.

#### **Research Gaps Matrix**

| Topic/Outcome | Synthetic Benchmarks | Real Game Maps | Multi-objective Settings | Evolutionary Comparison |
| ----- | ----- | ----- | ----- | ----- |
| High-Dimensional Continuous | **22** | **GAP** | **6** | **8** |
| High-Dimensional Combinatorial | **15** | **1** | **5** | **6** |
| Board Game Map Design | **GAP** | **2** | **GAP** | **2** |
| Procedural Content Generation | **GAP** | **3** | **GAP** | **3** |

**Figure undefined:** Matrix showing research coverage by topic/outcome versus study attribute.

### **Open Research Questions**

Future work should focus on bridging methodological advances with practical applications:

| Question | Why |
| ----- | ----- |
| **How do advanced Bayesian optimization algorithms perform on real-world board game map generation tasks?** | Direct empirical validation will clarify practical benefits over existing PCG approaches |
| **Can multi-objective Bayesian optimization efficiently balance competing gameplay criteria in map design?** | Board games require trade-offs between balance/fun/variety; multi-objective BO could automate this |
| **What hybrid frameworks best combine evolutionary/metaheuristic methods with modern Bayesian optimization?** | Hybridizing strengths may yield superior results for complex/highly-constrained content generation |

**Figure undefined:** Open questions highlighting future directions at the intersection of BO and procedural content generation.

In summary: **Bayesian optimization is a promising approach for high-dimensional combinatorial board game map design**, especially when leveraging recent algorithmic innovations tailored for such spaces—but more direct application studies are needed to fully realize its potential.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

Yuqian Gui, Dawei Zhan and Tianrui Li. "Taking another step: A simple approach to high-dimensional Bayesian optimization." *Inf. Sci.*, 679 (2024): 121056\. [https://doi.org/10.1016/j.ins.2024.121056](https://doi.org/10.1016/j.ins.2024.121056)

Dongyang Li and Dawei Zhan. "High-Dimensional Bayesian Optimization Based on Local and Global Models." *2025 7th International Conference on Data-driven Optimization of Complex Systems (DOCS)* (2025): 247-252. [https://doi.org/10.1109/docs67533.2025.11200954](https://doi.org/10.1109/docs67533.2025.11200954)

H. Bian, Jie Tian, Jialiang Yu and Han Yu. "Bayesian Co-evolutionary Optimization based entropy search for high-dimensional many-objective optimization." *Knowl. Based Syst.*, 274 (2023): 110630\. [https://doi.org/10.1016/j.knosys.2023.110630](https://doi.org/10.1016/j.knosys.2023.110630)

Leonard Papenmeier, Matthias Poloczek and Luigi Nardi. "Understanding High-Dimensional Bayesian Optimization." *ArXiv*, abs/2502.09198 (2025). [https://doi.org/10.48550/arxiv.2502.09198](https://doi.org/10.48550/arxiv.2502.09198)

Jundi Huang and Dawei Zhan. "An Adaptive Dropout Approach for High-Dimensional Bayesian Optimization." *ArXiv*, abs/2504.11353 (2025). [https://doi.org/10.48550/arxiv.2504.11353](https://doi.org/10.48550/arxiv.2504.11353)

Dawei Zhan. "Expected Coordinate Improvement for High-Dimensional Bayesian Optimization." *ArXiv*, abs/2404.11917 (2024). [https://doi.org/10.1016/j.swevo.2024.101745](https://doi.org/10.1016/j.swevo.2024.101745)

Lam Ngo, Huong Ha, Jeffrey Chan and Hongyu Zhang. "MOCA-HESP: Meta High-dimensional Bayesian Optimization for Combinatorial and Mixed Spaces via Hyper-ellipsoid Partitioning." *ArXiv*, abs/2508.06847 (2025). [https://doi.org/10.48550/arxiv.2508.06847](https://doi.org/10.48550/arxiv.2508.06847)

Satoshi Koide and Ayano Okoso. "High-Dimensional Slider-Based Preferential Bayesian Optimization With Mixed Local and Global Acquisition Strategies." *IEEE Access*, 12 (2024): 90862-90872. [https://doi.org/10.1109/access.2024.3421290](https://doi.org/10.1109/access.2024.3421290)

Ling Jin and Dawei Zhan. "An Adaptive Line Search Method for High-Dimensional Bayesian Optimization." *2025 7th International Conference on Data-driven Optimization of Complex Systems (DOCS)* (2025): 80-85. [https://doi.org/10.1109/docs67533.2025.11200541](https://doi.org/10.1109/docs67533.2025.11200541)

Leonard Papenmeier, Luigi Nardi and Matthias Poloczek. "Bounce: Reliable High-Dimensional Bayesian Optimization for Combinatorial and Mixed Spaces." \*\* (2023).

Vu Viet Hoang, H. Tran, Sunil Gupta and Vu Nguyen. "High Dimensional Bayesian Optimization using Lasso Variable Selection." *ArXiv*, abs/2504.01743 (2025). [https://doi.org/10.48550/arxiv.2504.01743](https://doi.org/10.48550/arxiv.2504.01743)

Hung The Tran, Sunil Gupta, Santu Rana and S. Venkatesh. "Trading Convergence Rate with Computational Budget in High Dimensional Bayesian Optimization." *ArXiv*, abs/1911.11950 (2019). [https://doi.org/10.1609/aaai.v34i03.5623](https://doi.org/10.1609/aaai.v34i03.5623)

Carlos Celemin. "Bayesian Optimization-based Search for Agent Control in Automated Game Testing." *2024 IEEE Conference on Games (CoG)* (2024): 1-4. [https://doi.org/10.1109/cog60054.2024.10645653](https://doi.org/10.1109/cog60054.2024.10645653)

Zhiyuan Miao and Dawei Zhan. "A Greedy Optimization Approach for High-Dimensional Bayesian Optimization." *2025 7th International Conference on Data-driven Optimization of Complex Systems (DOCS)* (2025): 112-117. [https://doi.org/10.1109/docs67533.2025.11200529](https://doi.org/10.1109/docs67533.2025.11200529)

Bahador Rashidi, Kerrick Johnstonbaugh and Chao Gao. "Cylindrical Thompson Sampling for High-Dimensional Bayesian Optimization." \*\* (2024): 3502-3510.

Jiaming Lu and Rong J.B. Zhu. "High dimensional Bayesian Optimization via Condensing-Expansion Projection." *ArXiv*, abs/2408.04860 (2024). [https://doi.org/10.48550/arxiv.2408.04860](https://doi.org/10.48550/arxiv.2408.04860)

K. Antonov, E. Raponi, Hao Wang and Carola Doerr. "High Dimensional Bayesian Optimization with Kernel Principal Component Analysis." \*\* (2022): 118-131. [https://doi.org/10.48550/arxiv.2204.13753](https://doi.org/10.48550/arxiv.2204.13753)

Aryan Deshwal, Sebastian Ament, M. Balandat, E. Bakshy, J. Doppa and David Eriksson. "Bayesian Optimization over High-Dimensional Combinatorial Spaces via Dictionary-based Embeddings." \*\* (2023): 7021-7039. [https://doi.org/10.48550/arxiv.2303.01774](https://doi.org/10.48550/arxiv.2303.01774)

Sam Daulton, David Eriksson, M. Balandat and E. Bakshy. "Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces." *ArXiv*, abs/2109.10964 (2021).

Riccardo Moriconi, M. Deisenroth and K. S. Sesh Kumar. "High-dimensional Bayesian optimization using low-dimensional feature spaces." *Machine Learning*, 109 (2019): 1925 \- 1943\. [https://doi.org/10.1007/s10994-020-05899-z](https://doi.org/10.1007/s10994-020-05899-z)

David Eriksson and M. Jankowiak. "High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces." *ArXiv*, abs/2103.00349 (2021).

E. Hellsten, Carl Hvarfner, Leonard Papenmeier and Luigi Nardi. "High-dimensional Bayesian Optimization with Group Testing." *ArXiv*, abs/2310.03515 (2023). [https://doi.org/10.48550/arxiv.2310.03515](https://doi.org/10.48550/arxiv.2310.03515)

Xingchen Wan, Vu Nguyen, Huong Ha, Binxin Ru, Cong Lu and Michael A. Osborne. "Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces." \*\* (2021): 10663-10674.

Carl Hvarfner, E. Hellsten and Luigi Nardi. "Vanilla Bayesian Optimization Performs Great in High Dimensions." *ArXiv*, abs/2402.02229 (2024).

Yiqi Liu and Dawei Zhan. "Adaptive Subspace Selection Approach for High-Dimensional Bayesian Optimization." *2025 7th International Conference on Data-driven Optimization of Complex Systems (DOCS)* (2025): 241-246. [https://doi.org/10.1109/docs67533.2025.11200884](https://doi.org/10.1109/docs67533.2025.11200884)

Lam Ngo, Huong Ha, Jeffrey Chan, Vu Nguyen and Hongyu Zhang. "High-dimensional Bayesian Optimization via Covariance Matrix Adaptation Strategy." *Trans. Mach. Learn. Res.*, 2024 (2024). [https://doi.org/10.48550/arxiv.2402.03104](https://doi.org/10.48550/arxiv.2402.03104)

Sasan Amini, Inneke Vannieuwenhuyse and Alejandro Morales-Hernández. "Constrained Bayesian Optimization: A Review." *IEEE Access*, 13 (2025): 1581-1593. [https://doi.org/10.1109/access.2024.3522876](https://doi.org/10.1109/access.2024.3522876)

Changyong Oh, J. Tomczak, E. Gavves and M. Welling. "Combinatorial Bayesian Optimization using the Graph Cartesian Product." \*\* (2019): 2910-2920.

Luiz Jonatã Pires de Araújo, Alexandr Grichshenko, R. Pinheiro, R. D. Saraiva and Susanna Gimaeva. "Map Generation and Balance in the Terra Mystica Board Game Using Particle Swarm and Local Search." *Advances in Swarm Intelligence*, 12145 (2020): 163 \- 175\. [https://doi.org/10.1007/978-3-030-53956-6\_15](https://doi.org/10.1007/978-3-030-53956-6_15)

Zhitong Xu and Shandian Zhe. "Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization." *ArXiv*, abs/2402.02746 (2024). [https://doi.org/10.48550/arxiv.2402.02746](https://doi.org/10.48550/arxiv.2402.02746)

E. Hellsten, Carl Hvarfner, Leonard Papenmeier and Luigi Nardi. "Leveraging Axis-Aligned Subspaces for High-Dimensional Bayesian Optimization with Group Testing." *ArXiv*, abs/2504.06111 (2025). [https://doi.org/10.48550/arxiv.2504.06111](https://doi.org/10.48550/arxiv.2504.06111)

Lam Ngo, Huong Ha, Jeffrey Chan and Hongyu Zhang. "BOIDS: High-Dimensional Bayesian Optimization via Incumbent-Guided Direction Lines and Subspace Embeddings." \*\* (2025): 19659-19667. [https://doi.org/10.1609/aaai.v39i18.34165](https://doi.org/10.1609/aaai.v39i18.34165)

M. Santoni, E. Raponi, R. Leone and Carola Doerr. "Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB." *ACM Transactions on Evolutionary Learning*, 4 (2023): 1 \- 33\. [https://doi.org/10.1145/3670683](https://doi.org/10.1145/3670683)

R. Priem, Y. Diouane, Nathalie Bartoli, S. Dubreuil and Paul Maxime Valentin Saves. "High-Dimensional Bayesian Optimization Using Both Random and Supervised Embeddings." *ArXiv*, abs/2502.00854 (2024). [https://doi.org/10.2514/1.j063488](https://doi.org/10.2514/1.j063488)

Zan Shen, Xiaochun Li and Junfa Mao. "An Efficient Method for Ball Map Design Based on Bayesian Optimization." *2021 International Conference on Microwave and Millimeter Wave Technology (ICMMT)* (2021): 1-3. [https://doi.org/10.1109/icmmt52847.2021.9618132](https://doi.org/10.1109/icmmt52847.2021.9618132)

Lei Song, Ke Xue, Xiaobin Huang and Chaojun Qian. "Monte Carlo Tree Search based Variable Selection for High Dimensional Bayesian Optimization." *ArXiv*, abs/2210.01628 (2022). [https://doi.org/10.48550/arxiv.2210.01628](https://doi.org/10.48550/arxiv.2210.01628)

Miguel González Duque, Rasmus Berg Palm, David Ha and S. Risi. "Finding Game Levels with the Right Difficulty in a Few Trials through Intelligent Trial-and-Error." *2020 IEEE Conference on Games (CoG)* (2020): 503-510. [https://doi.org/10.1109/cog47356.2020.9231548](https://doi.org/10.1109/cog47356.2020.9231548)

\---

# **Evidence suggests distance-weighted gravity models are generally effective proxies for resource accessibility in discrete spatial networks, but their accuracy depends on model specification and context.**

## **1\. Introduction**

Distance-weighted gravity models are widely used to estimate resource accessibility in discrete spatial networks, such as healthcare, public transport, parks, and other urban services. These models account for both the supply of resources and the deterrence effect of distance or travel time, making them more nuanced than simple proximity or provider-to-population ratios (Stacherl and Sauzet 2023; Sun et al. 2024; Begicheva and Begicheva 2025; Pecoraro et al. 2024; Delamater 2013, 30-43 ; Zhu et al. 2025; Yang et al. 2019; Piovani et al. 2018; Yiqiu et al. 2025; Shi et al. 2022, 265 \- 289). Recent systematic reviews and empirical studies highlight that gravity-based approaches—especially those incorporating network-based travel times and facility capacities—provide interpretable and policy-relevant measures of potential access (Stacherl and Sauzet 2023; Sun et al. 2024; Begicheva and Begicheva 2025; Pecoraro et al. 2024). However, their effectiveness as proxies depends on careful calibration of distance decay functions, consideration of supply-demand competition, and sensitivity to spatial configuration (Sun et al. 2024; Begicheva and Begicheva 2025; Pecoraro et al. 2024; Xia et al. 2018). While gravity models are dominant in methodological development and application, especially in healthcare and urban planning contexts (Stacherl and Sauzet 2023), limitations remain regarding parameter selection, data aggregation effects, and context-specific dynamics (Sun et al. 2024; Begicheva and Begicheva 2025; Pecoraro et al. 2024). Overall, the literature supports the use of distance-weighted gravity models as effective proxies for resource accessibility in discrete spatial networks when appropriately specified.

**Figure 1:** Overall research support for gravity-based accessibility

## **2\. Methods**

A comprehensive search was conducted across over 170 million research papers using Consensus, including sources such as Semantic Scholar and PubMed. The search strategy involved multiple targeted queries covering foundational theories, methodological critiques, domain applications (healthcare, parks, transport), parameter sensitivity analyses, and comparative studies with alternative models. In total, 990 papers were identified; after screening and eligibility assessment, 50 highly relevant papers were included in this review.

| Identification | Screening | Eligibility | Included |
| ----- | ----- | ----- | ----- |
| 990 | 626 | 424 | 50 |

**Figure 2:** Flow diagram of paper selection process for this review. Eight unique search groups were used to ensure broad coverage of theoretical foundations, methodological diversity, domain-specific applications, parameter sensitivity, and adjacent constructs.

## **3\. Results**

### **3.1 Dominance of Gravity Models in Accessibility Research**

Gravity-based models—especially the Two-Step Floating Catchment Area (2SFCA) family—are the most commonly used methods for measuring potential spatial access to resources such as healthcare facilities (Stacherl and Sauzet 2023), public transport (Yang et al. 2019), parks (Yiqiu et al. 2025), jobs (Hu and Downs 2019), and green spaces (Zhu et al. 2025). Methodological developments have focused on refining these models by introducing variable catchment sizes, distance decay functions, supply-demand competition adjustments, multi-modal travel modes, and temporal dynamics (Stacherl and Sauzet 2023; Sun et al. 2024; Delamater 2013, 30-43 ).

### **3.2 Empirical Performance & Validation**

Empirical studies demonstrate that gravity-based accessibility indices correspond moderately well with observed utilization patterns (e.g., R² ≈ 0.25 for clinic visits; MAPE \< 28%) when calibrated with real network travel times and facility capacities (Sun et al. 2024; Roy and Rai 2023). Gravity models outperform simple proximity or ratio methods by capturing cross-boundary flows and competitive effects among facilities (Schuurman et al. 2010, 29-45; Talen and Anselin 1998, 595 \- 613). In urban planning contexts (parks/green spaces), gravity-based measures reveal more nuanced spatial disparities than traditional methods (Zhu et al. 2025; Yiqiu et al. 2025).

### **3.3 Key Model Parameters & Sensitivities**

The effectiveness of gravity models is highly sensitive to:

* The choice of distance metric (network vs. Euclidean) (Roy and Rai 2023; Frew et al. 2017, 128-142)- The form and calibration of the distance decay function (Sun et al. 2024; Li et al. 2019)- Inclusion of facility capacity/supply-demand competition (Sun et al. 2024; Pecoraro et al. 2024)- Spatial scale/zoning effects (modifiable areal unit problem) (Bryant and Delamater 2019, 219 \- 229)Recent advances include integrating real-time travel data (Li et al. 2022), multi-modal transport options (Sun et al. 2024), hierarchical facility structures (Tao et al. 2020), and dynamic temporal factors (Järv et al. 2018).

### **3.4 Limitations & Comparative Findings**

While generally effective as proxies for accessibility:

* Gravity models may overestimate access if not properly calibrated or if geographic weights are misapplied (Zhang 2021).  
* They can be less suitable in contexts with strong temporal variability or unique system dynamics (e.g., multi-island states) unless extended with additional parameters or hybridized with other approaches (e.g., radiation models) (Martin and Paul 2022).  
* Deep learning approaches may further improve estimates where partial utilization data is available but require more complex implementation (Mishina et al. 2024).

#### **Results Timeline**

* **2006**  
  * 1 paper: (Yang et al. 2006, 23-32)- **2013**  
  * 1 paper: (Delamater 2013, 30-43 )- **2017**  
  * 1 paper: (Wu et al. 2017, 45-54)- **2018**  
  * 2 papers: (Piovani et al. 2018; Xia et al. 2018)- **2019**  
  * 3 papers: (Yang et al. 2019; Li et al. 2019; Chang et al. 2019)- **2020**  
  * 1 paper: (Cooper and Chiaradia 2020, 100525)- **2022**  
  * 3 papers: (Shi et al. 2022, 265 \- 289; Liu and Zlatanova 2022, 3277 \- 3294; Li et al. 2022)- **2023**  
  * 2 papers: (Stacherl and Sauzet 2023; Roy and Rai 2023)- **2024**  
  * 3 papers: (Sun et al. 2024; Pecoraro et al. 2024; Dabagh et al. 2024, 359 \- 369)- **2025**  
  * 3 papers: (Begicheva and Begicheva 2025; Zhu et al. 2025; Yiqiu et al. 2025\)**Figure 3:** Chronological distribution of key studies on gravity-based accessibility modeling. Larger markers indicate more citations.

#### **Top Contributors**

| Type | Name | Papers |
| ----- | ----- | ----- |
| Author | P. Delamater | (Delamater 2013, |

     30-43

; Bryant and Delamater 2019, 219 \- 229)| | Author | Yanfang Liu | (Yang et al. 2019)| | Author | Neng Wan | (Wan et al. 2012, 1073 \- 1089)| | Journal | *Health & place* | (Delamater 2013, 30-43 ; Luo and Qi 2009, 1100-7 )| | Journal | *Scientific Reports* | (Yiqiu et al. 2025; Li et al. 2019)| | Journal | *Sustainability* | (Yang et al. 2019; Li et al. 2022)|

**Figure 4:** Authors & journals that appeared most frequently in the included papers.

## **4\. Discussion**

The literature strongly supports the use of distance-weighted gravity models as effective proxies for resource accessibility in discrete spatial networks across domains such as healthcare planning, public transport evaluation, park/green space equity analysis, and job access studies (Stacherl and Sauzet 2023; Sun et al. 2024; Zhu et al. 2025; Yang et al. 2019). Their main strengths lie in their ability to jointly represent supply availability and demand competition while accounting for realistic travel impedance via network distances or times.

However, their validity is contingent upon careful model specification: using network-based distances rather than straight-line measures improves realism; calibrating distance decay parameters enhances predictive accuracy; incorporating facility capacity prevents overestimation; and addressing scale/zoning issues mitigates aggregation bias (Sun et al. 2024; Roy and Rai 2023; Bryant and Delamater 2019, 219 \- 229). Methodological innovations continue to address these challenges by integrating multi-modal travel options or dynamic temporal factors.

Despite these advances—and widespread empirical adoption—gravity models remain proxies: they estimate potential rather than realized access/utilization. Their performance can be limited by data quality (e.g., incomplete road networks), unmodeled behavioral factors (e.g., user preferences), or context-specific system dynamics (e.g., islands with traveling doctors) that require further methodological adaptation or hybridization with other modeling frameworks (e.g., radiation/deep learning) (Martin and Paul 2022; Mishina et al. 2024).

### **Claims and Evidence Table**

| Claim | Evidence Strength | Reasoning | Papers |
| ----- | ----- | ----- | ----- |
| Gravity models are widely used/effective proxies for resource accessibility | Evidence strength: Strong (8/10) | Systematic reviews show dominance in empirical research; moderate-to-high correspondence with observed patterns | (Stacherl and Sauzet 2023; Sun et al. 2024; Zhu et al. 2025; Yang et al. 2019\) |
| Network-based distances improve realism over Euclidean distances | Evidence strength: Moderate (7/10) | Empirical studies show better fit/calibration when using actual travel times/distances | (Roy and Rai 2023; Frew et al. 2017, 128-142) |
| Model performance is sensitive to decay function/parameter choices | Evidence strength: Moderate (6/10) | Calibration studies show significant impact on results; improper choices can distort access estimates | (Sun et al. 2024; Li et al. 2019\) |
| Facility capacity/competition must be included for accurate results | Evidence strength: Moderate (6/10) | Models omitting supply-demand constraints overestimate access; improved versions address this | (Sun et al. 2024; Pecoraro et al. 2024\) |
| Gravity models may misrepresent access if weights/aggregation are mishandled | Evidence strength: Weak (3/10) | Studies highlight risks from inappropriate weighting/geographic aggregation | (Bryant and Delamater 2019, 219 \- 229; Zhang 2021\) |
| Hybrid/deep learning approaches may outperform traditional gravity models where partial utilization data exist | Evidence strength: Weak (2/10) | Early evidence suggests improved accuracy but requires complex implementation | (Mishina et al. 2024\) |

**Figure 5:** Key claims and support evidence identified in these papers.

## **5\. Conclusion**

Distance-weighted gravity models are empirically validated as effective proxies for resource accessibility in discrete spatial networks when carefully specified—using network-based distances/times, calibrated decay functions, supply-demand constraints—and remain dominant tools across multiple domains. However, their proxy nature means they estimate potential rather than realized access/utilization; ongoing methodological refinements continue to address known limitations.

### **Research Gaps**

| Topic/Outcome | Healthcare Access | Park/Green Space Access | Public Transport Access | Job Access |
| ----- | ----- | ----- | ----- | ----- |
| Urban Networks | **18** | **8** | **10** | **5** |
| Rural/Peripheral | **6** | **1** | **GAP** | **GAP** |
| Multi-modal Analysis | **5** | **GAP** | **7** | **GAP** |
| Temporal Dynamics | **3** | **GAP** | **1** | **1** |

**Figure 6:** Matrix showing concentration of research by topic/outcome versus study attribute.

### **Open Research Questions**

Future research should focus on improving behavioral realism (actual utilization vs potential), integrating dynamic temporal/spatial factors at finer scales, adapting methods to unique system contexts (multi-island states), and leveraging new data sources/methods.

| Question | Why |
| ----- | ----- |
| **How do different forms/calibrations of distance decay affect the validity of gravity-based accessibility measures across diverse network types?** | Decay function choice critically shapes results; systematic comparison across settings would guide best practices. |
| **Can hybrid or deep learning approaches significantly outperform traditional gravity models for predicting realized resource utilization?** | Early evidence suggests promise but lacks large-scale validation; could bridge gap between potential/proxy measures and actual use. |
| **What adaptations are needed for gravity-based accessibility metrics to accurately reflect access in highly dynamic or fragmented systems?** | Unique contexts like multi-island states challenge standard assumptions; tailored solutions could improve equity/resource allocation. |

**Figure 7:** Open questions highlighting future directions for research on gravity-model-based accessibility.

In summary: Distance-weighted gravity models remain robust proxies for resource accessibility in discrete spatial networks when carefully specified—but continued refinement is needed to address behavioral realism and context-specific challenges.

*These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at [https://consensus.app](https://consensus.app). © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent.*

## **References**

B. Stacherl and Odile Sauzet. "Gravity models for potential spatial healthcare access measurement: a systematic methodological review." *International Journal of Health Geographics*, 22 (2023). [https://doi.org/10.1186/s12942-023-00358-z](https://doi.org/10.1186/s12942-023-00358-z)

Shijie Sun, Qun Sun, Fubing Zhang and Jingzhen Ma. "A Spatial Accessibility Study of Public Hospitals: A Multi-Mode Gravity-Based Two-Step Floating Catchment Area Method." *Applied Sciences* (2024). [https://doi.org/10.3390/app14177713](https://doi.org/10.3390/app14177713)

S. Begicheva and A. Begicheva. "Modified gravity model for assessing healthcare accessibility: problem statement, algorithm, and implementation." *Journal Of Applied Informatics* (2025). [https://doi.org/10.37791/2687-0649-2025-20-5-4-21](https://doi.org/10.37791/2687-0649-2025-20-5-4-21)

F. Pecoraro, Marco Cellini, D. Luzi and Fabrizio Clemente. "Analysing the intra and interregional components of spatial accessibility gravity model to capture the level of equity in the distribution of hospital services in Italy: do they influence patient mobility?." *BMC Health Services Research*, 24 (2024). [https://doi.org/10.1186/s12913-024-11411-3](https://doi.org/10.1186/s12913-024-11411-3)

P. Delamater. "Spatial accessibility in suboptimally configured health care systems: a modified two-step floating catchment area (M2SFCA) metric.." *Health & place*, 24 (2013): 30-43. [https://doi.org/10.1016/j.healthplace.2013.07.012](https://doi.org/10.1016/j.healthplace.2013.07.012)

Yifanzi Zhu, Qiuyi Yang, Shuying Guo, Yuhan Wen, Xinyi Wang and Rui Wang. "Modeling Urban Green Access: Combining Zone-Based Proximity and Demand-Weighted Metrics in a Medium-Sized U.S. City." *Land* (2025). [https://doi.org/10.3390/land14091926](https://doi.org/10.3390/land14091926)

Ruqin Yang, Yao Liu, Yanfang Liu, Hui Liu and W. Gan. "Comprehensive Public Transport Service Accessibility Index—A New Approach Based on Degree Centrality and Gravity Model." *Sustainability* (2019). [https://doi.org/10.3390/su11205634](https://doi.org/10.3390/su11205634)

Duccio Piovani, E. Arcaute, Gabriela Uchoa, A. Wilson and M. Batty. "Measuring accessibility using gravity and radiation models." *Royal Society Open Science*, 5 (2018). [https://doi.org/10.1098/rsos.171668](https://doi.org/10.1098/rsos.171668)

Zuo Yiqiu, Xinlei Ding, Yajuan Wei, Wenguo Wang and Mingjing Wang. "GIS-based accessibility analysis of urban park green space landscape." *Scientific Reports*, 15 (2025). [https://doi.org/10.1038/s41598-025-13750-5](https://doi.org/10.1038/s41598-025-13750-5)

Yiling Shi, J. Yang, M. Keith, Kun Song, Ying Li and Chenghe Guan. "Spatial Accessibility Patterns to Public Hospitals in Shanghai: An Improved Gravity Model." *The Professional Geographer*, 74 (2022): 265 \- 289\. [https://doi.org/10.1080/00330124.2021.2000445](https://doi.org/10.1080/00330124.2021.2000445)

Nan Xia, Liang Cheng, Song Chen, Xiaoyan Wei, Wenwen Zong and Manchun Li. "Accessibility based on Gravity-Radiation model and Google Maps API: A case study in Australia." *Journal of Transport Geography* (2018). [https://doi.org/10.1016/j.jtrangeo.2018.09.009](https://doi.org/10.1016/j.jtrangeo.2018.09.009)

Shabnam Dabagh, Lory Michelle Bresciani Miristice and Guido Gentile. "Accessibility Via Public Transport Through Gravity Models Based on Open Data." *Transport and Telecommunication Journal*, 25 (2024): 359 \- 369\. [https://doi.org/10.2478/ttj-2024-0026](https://doi.org/10.2478/ttj-2024-0026)

Sara M. Roy and Anupma Rai. "The Quest for Quality Healthcare Services – What we can learn from a settlement-based case study?." *IOP Conference Series: Earth and Environmental Science*, 1164 (2023). [https://doi.org/10.1088/1755-1315/1164/1/012017](https://doi.org/10.1088/1755-1315/1164/1/012017)

Zhe Li, Tao Ren, Xiaoqi Ma, Simiao Liu, Yixin Zhang and Tao Zhou. "Identifying influential spreaders by gravity model." *Scientific Reports*, 9 (2019). [https://doi.org/10.1038/s41598-019-44930-9](https://doi.org/10.1038/s41598-019-44930-9)

Liu Liu and S. Zlatanova. "A spatial data model for accessibility with indoor temporal changes." *Transactions in GIS*, 26 (2022): 3277 \- 3294\. [https://doi.org/10.1111/tgis.13002](https://doi.org/10.1111/tgis.13002)

Crispin H. V. Cooper and A. Chiaradia. "sDNA: 3-d spatial network analysis for GIS, CAD, Command Line & Python." *SoftwareX*, 12 (2020): 100525\. [https://doi.org/10.1016/j.softx.2020.100525](https://doi.org/10.1016/j.softx.2020.100525)

Chao Wu, X. Ye, Qingyun Du and P. Luo. "Spatial effects of accessibility to parks on housing prices in Shenzhen, China." *Habitat International*, 63 (2017): 45-54. [https://doi.org/10.1016/j.habitatint.2017.03.010](https://doi.org/10.1016/j.habitatint.2017.03.010)

Z. Chang, Jiayu Chen, Weifeng Li and Xin Li. "Public transportation and the spatial inequality of urban park accessibility: New evidence from Hong Kong." *Transportation Research Part D: Transport and Environment* (2019). [https://doi.org/10.1016/j.trd.2019.09.012](https://doi.org/10.1016/j.trd.2019.09.012)

Duck-Hye Yang, Robert M. Goerge and R. Mullner. "Comparing GIS-Based Methods of Measuring Spatial Accessibility to Health Services." *Journal of Medical Systems*, 30 (2006): 23-32. [https://doi.org/10.1007/s10916-006-7400-5](https://doi.org/10.1007/s10916-006-7400-5)

Juchen Li, Xiucheng Guo, Ruiying Lu and Yibang Zhang. "Analysing Urban Tourism Accessibility Using Real-Time Travel Data: A Case Study in Nanjing, China." *Sustainability* (2022). [https://doi.org/10.3390/su141912122](https://doi.org/10.3390/su141912122)

W. Luo and Y. Qi. "An enhanced two-step floating catchment area (E2SFCA) method for measuring spatial accessibility to primary care physicians.." *Health & place*, 15 4 (2009): 1100-7. [https://doi.org/10.1016/j.healthplace.2009.06.002](https://doi.org/10.1016/j.healthplace.2009.06.002)

N. Schuurman, Myriam Bérubé and Valorie A. Crooks. "Measuring potential spatial access to primary health care physicians using a modified gravity model." *Canadian Geographer*, 54 (2010): 29-45. [https://doi.org/10.1111/j.1541-0064.2009.00301.x](https://doi.org/10.1111/j.1541-0064.2009.00301.x)

E. Talen and L. Anselin. "Assessing Spatial Equity: An Evaluation of Measures of Accessibility to Public Playgrounds." *Environment and Planning A*, 30 (1998): 595 \- 613\. [https://doi.org/10.1068/a300595](https://doi.org/10.1068/a300595)

James Bryant and P. Delamater. "Examination of spatial accessibility at micro- and macro-levels using the enhanced two-step floating catchment area (E2SFCA) method." *Annals of GIS*, 25 (2019): 219 \- 229\. [https://doi.org/10.1080/19475683.2019.1641553](https://doi.org/10.1080/19475683.2019.1641553)

Roxanne Brizan-St. Martin and Juel Paul. "Evaluating the Performance of GIS Methodologies for Quantifying Spatial Accessibility to Healthcare in Multi-Island Micro States (MIMS).." *Health policy and planning* (2022). [https://doi.org/10.1093/heapol/czac001](https://doi.org/10.1093/heapol/czac001)

Lina Zhang. "Trap of weights: The reuse of weights in the floating catchment area (FCA) methods to measuring accessibility." *F1000Research*, 10 (2021). [https://doi.org/10.12688/f1000research.51483.2](https://doi.org/10.12688/f1000research.51483.2)

Zhuolin Tao, Yang Cheng and Jixiang Liu. "Hierarchical two-step floating catchment area (2SFCA) method: measuring the spatial accessibility to hierarchical healthcare facilities in Shenzhen, China." *International Journal for Equity in Health*, 19 (2020). [https://doi.org/10.1186/s12939-020-01280-7](https://doi.org/10.1186/s12939-020-01280-7)

Neng Wan, Bin Zou and Troy Sternberg. "A three-step floating catchment area method for analyzing spatial access to health services." *International Journal of Geographical Information Science*, 26 (2012): 1073 \- 1089\. [https://doi.org/10.1080/13658816.2011.624987](https://doi.org/10.1080/13658816.2011.624987)

Yujie Hu and J. Downs. "Measuring and Visualizing Place-Based Space-Time Job Accessibility." *ArXiv*, abs/2006.00268 (2019). [https://doi.org/10.1016/j.jtrangeo.2018.12.002](https://doi.org/10.1016/j.jtrangeo.2018.12.002)

R. Frew, G. Higgs, J. Harding and M. Langford. "Investigating geospatial data usability from a health geography perspective using sensitivity analysis: The example of potential accessibility to primary healthcare." *Journal of transport and health*, 6 (2017): 128-142. [https://doi.org/10.1016/j.jth.2017.03.013](https://doi.org/10.1016/j.jth.2017.03.013)

O. Järv, H. Tenkanen, M. Salonen, R. Ahas and T. Toivonen. "Dynamic cities: Location-based accessibility modelling as a function of time." *Applied Geography* (2018). [https://doi.org/10.1016/j.apgeog.2018.04.009](https://doi.org/10.1016/j.apgeog.2018.04.009)

Margarita E. Mishina, Sergey Mityagin, Alexander B. Belyi, Alexander A. Khrulkov and Stanislav Sobolevsky. "Towards Urban Accessibility: Modeling Trip Distribution to Assess the Provision of Social Facilities." *Smart Cities* (2024). [https://doi.org/10.3390/smartcities7050106](https://doi.org/10.3390/smartcities7050106)

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

