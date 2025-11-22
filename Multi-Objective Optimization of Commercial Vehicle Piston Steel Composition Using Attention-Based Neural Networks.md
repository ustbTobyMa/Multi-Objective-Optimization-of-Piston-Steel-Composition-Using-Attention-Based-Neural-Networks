***\*Multi-Objective Optimization of Commercial Vehicle Piston Steel Composition Using Attention-Based Neural Networks\****

Weitao Ma1, Yanjun Rao1, Zheyue Zhang1, Zijian Zhang1, Shuai Zhao1, Renbo Song1, Guanwen Dai2 Yongjin Wang1, Zengjian Feng3, 

1 School of Materials Science and Engineering, University of Science and Technology Beijing, Beijing 100083, P.R. China

2 HBIS Group ShiSteel Company, Shijiazhuang, 050031, Hebei, P.R. China

3 Shandong Binzhou Bohai Piston Co., Ltd., Binzhou 256602, P.R. China

 Corresponding author. E-mail address: songrb@mater.ustb.edu.cn (Renbo Song)

***\*Abstract\****: 

Highlights:
- Interpretable attention‑based surrogate + NSGA‑II yields weldable, cost‑capped piston steels.
- Constrained Pareto front exposes high‑performance, balanced and cost‑sensitive design zones.
- Two AI‑designed steels (QT/NQT) validated; 18–25% strength gains within ≤30% cost cap.
- Improved oxidation resistance at 600 °C/500 h with dense, adherent oxide scales.
- Framework generalises to multi‑objective alloy design with uncertainty‑aware ranking.

Commercial vehicle piston steels must simultaneously deliver high strength, toughness, and high‑temperature durability at acceptable cost, yet empirical development scales poorly in multi‑component spaces and offers limited interpretability. This study aimed to develop an interpretable, attention‑based deep learning framework coupled with constrained multi‑objective optimization to generate manufacturable steel compositions and to validate representative candidates under standard protocols. A harmonized composition–process–property dataset (n=2,847) was assembled, and a calibrated multi‑task model was trained to predict key properties while indicating which elements and heat‑treatment variables matter most for each target. The search then used a genetic multi‑objective algorithm within industrial composition/processing bounds, enforcing hard constraints of carbon equivalent (CEV) ≤ 0.60 and relative cost ≤ +30% versus Piston Steel 4140 for commercial vehicles; uncertainty‑aware ranking and repeated runs established robustness. The constrained Pareto front revealed high‑performance, balanced, and cost‑effective decision zones and a clear knee point. Two steels were produced (quenched–tempered and non‑quenched–tempered). Under weldability and cost constraints, both AI‑designed steels showed significant improvements in mechanical and physical properties; oxidation resistance and strength at 600 °C were also improved. These results demonstrate that calibrated interpretability combined with constrained optimization yields weldable, cost‑compatible piston‑steel designs and provides a reproducible, transferable pathway from data to deployment.

**Keywords: Steel composition optimization, Multi-objective optimization, Attention mechanism, Materials informatics, Machine learning, Piston steel, Genetic algorithm, Interpretable AI**

 

***\*1 Introduction\****

Commercial vehicle pistons operate under combined thermomechanical loads that are severe and sustained. Peak in‑cylinder pressures and temperatures, rapid thermal cycling, lubricant and combustion product exposure, and long service lifetimes jointly impose stringent requirements on piston steels. In practice, designers must balance strength and toughness at room temperature, ensure durability at elevated temperatures, and control density, thermal transport and total cost, all within melting, heat‑treatment and inspection windows. Meeting these concurrent targets remains a central challenge for heavy‑duty powertrains as efficiency and emissions regulations tighten and duty cycles diversify [1–5].

Traditional alloy development for piston steels has relied on empirical rules, incremental compositional variations around established grades, and focused testing campaigns[1]. While such practice has delivered robust grades for historical duty cycles, it scales poorly when the design space expands to multi‑component alloys with non‑linear interactions and when multiple, partially conflicting objectives must be satisfied simultaneously. Factorial experiments become prohibitive; extrapolating from narrow windows risks missing synergistic or antagonistic interactions; and trial‑and‑error exploration under real manufacturing constraints is costly and slow. Moreover, classical single‑objective optimization does not expose the inherent trade‑offs among strength, toughness, high‑temperature stability, and cost, which must be weighed explicitly by designers and procurement teams. Practically, design decisions must respect weldability and cost caps under industrial windows, turning alloy selection into a constrained, multi‑objective trade‑off that benefits from interpretable, data‑driven support[4–5].

Data‑driven methods have emerged as complementary tools for materials design. Supervised learning can interpolate composition–process–property relationships when supported by sufficient data, and inverse design strategies can propose candidates consistent with learned patterns [7–12]. However, two persistent gaps limit practical impact. The first is interpretability: many high‑performing models behave as black boxes, providing limited insight into which features control specific targets and why. For metallurgists and manufacturing engineers, this opacity complicates validation, risk assessment, and translation to production windows. The second is decision support under multiple objectives and constraints: proposing a single "optimal" composition obscures admissible trade‑offs, while ignoring feasibility or uncertainty can lead to brittle recommendations. These gaps are amplified when targets are partly scenario‑dependent (e.g., high‑temperature strength, oxidation), because operating conditions modulate responses that are not purely intrinsic to composition and heat treatment[6–10].

Attention mechanisms, originally developed in sequence modeling, have shown promise for tabular scientific data by learning context‑dependent feature weights that can be inspected post hoc. When embedded in multi‑task architectures, attention can share information across related targets while exposing property‑specific importance patterns, offering a bridge between statistical learning and mechanistic reasoning. In parallel, evolutionary multi‑objective algorithms such as NSGA‑II provide principled ways to approximate Pareto fronts under box constraints and domain‑specific feasibility rules, allowing designers to navigate performance–cost trade‑offs with explicit diversity. Yet, integrating interpretable prediction with constrained multi‑objective search and validating down‑selected candidates under industrially compatible processes remains underexplored in alloy systems[11–15].

This study addressed these needs by developing an interpretable, attention‑based deep learning framework for piston steel composition design and coupling it with NSGA‑II under realistic metallurgical constraints. A harmonized composition–process–property database was assembled from peer‑reviewed literature, industrial sources, and proprietary tests, with schema unification and quality controls to enable supervised learning. A dual‑mode input strategy was adopted to separate intrinsic descriptors (composition and heat treatment) from operating‑condition variables supplied only for scenario‑dependent targets, preserving clarity of scope for room‑temperature properties while enabling context for high‑temperature or cyclic responses. The predictive core comprised a multi‑head attention module integrated into a multi‑task network, producing target‑wise outputs and attention‑based feature attributions. The inverse design layer used NSGA‑II to generate diverse, feasible solutions that balanced performance against cost within process windows and compositional bounds relevant to production. Representative candidates were then selected for laboratory validation under standard protocols, with microstructural and property measurements acquired to assess agreement with the model and to evaluate manufacturability within industrially compatible ranges[16–18].

The present work was guided by three questions. First, can an attention‑based, multi‑task architecture learn composition–process–property relationships for piston steels with sufficient calibration to support surrogate‑based optimization, while providing feature‑level interpretability that aligns with metallurgical expectations? Second, can a constrained, multi‑objective search over composition and process variables recover a structured Pareto front that exposes actionable trade‑offs among targeted properties and cost, with adequate diversity for portfolio‑style decision‑making? Third, under laboratory processes bounded by industrial windows, do down‑selected candidates exhibit microstructure and properties consistent with the design intent, thereby supporting translation from data‑driven proposals to practical alloying and heat treatment routes?

Methodologically, this work contributes an integrated pathway that couples a transparent prediction layer with a decision layer and a validation layer, each aligned with manufacturing realities. The approach emphasizes calibrated prediction, interpretable attribution, constrained search, and targeted validation rather than unconstrained global optimization or opaque black‑box recommendations. Practically, the framework is designed to support existing alloy design workflows by surfacing diverse candidates that respect trace‑element limits, aggregate alloy caps, and process feasibility, and by providing human‑readable signals—via attention layouts and composition distributions—that can be weighed alongside procurement, sustainability, and quality considerations.

In what follows, the database and preprocessing protocol, attention‑based architecture, and multi‑objective optimization setup are described (Section 2). Predictive performance, interpretability outputs, Pareto fronts, and laboratory measurements under standard methods are reported (Section 3). The implications for alloy design, microstructure–property relations, manufacturability, and future research are discussed (Section 4). The overarching aim is to provide a tractable, interpretable, and experimentally grounded route from data to design for commercial vehicle piston steels, with generalizable elements applicable to other multi‑objective alloy systems.

***\*2 Methods\****

***\*2.1 Dataset Compilation and Preprocessing\****

A composition–process–property database comprising 2,847 unique piston steel records was compiled from peer‑reviewed literature, industrial databases, and proprietary testing records. Collection and curation were agent‑assisted (automated retrieval, deduplication, and schema harmonization), with human verification at each step[8,9,11]. Each entry contained chemical composition (wt%), heat‑treatment variables, optional operating conditions, and one or more target properties. Units and nomenclature were harmonized to ASTM/ISO conventions (e.g., MPa for strength, °C for temperature, wt% for composition). Records with incompatible or ambiguous standards were excluded before preprocessing. To balance intrinsic material response learning with service‑dependent predictions, a dual‑mode feature strategy was adopted. Core intrinsic inputs (16 features) included 12 alloying elements (C, Cr, Mo, Mn, Si, Ni, P, S, V, Ti, Al, Cu) and 4 heat‑treatment parameters (Quench_Temp, Temper_Temp, Cooling_Rate, Holding_Time). Scenario conditioning (Operating_Temp) was supplied only for targets known to exhibit operating‑environment dependence (e.g., high‑temperature tensile strength). This organization mirrored the Results section, where intrinsic and scenario‑dependent outcomes were reported separately.

Missing values (<3% overall) were imputed using the median within composition clusters derived by k‑means (k = 50) to preserve local alloying context. Outliers were filtered via an isolation forest (contamination = 0.05), removing approximately 5% of entries that violated metallurgical plausibility or conflicted with stated testing standards. Robust scaling was applied feature‑wise; for heavy‑tailed properties (e.g., high‑temperature tensile strength), quantile normalization was additionally used. Train/validation/test splits (70/15/15) were stratified by source and steel category to reduce distribution shift and prevent information leakage across splits. Metadata for standards (e.g., tensile: ASTM E8/E8M; impact: ASTM E23; high‑temperature tensile: ASTM E21) and specimen geometry were checked for consistency and recorded for downstream interpretation.

High‑temperature tensile target definition (dataset construction). For each composition/process entry with elevated‑temperature tests, we recorded the test temperature (Operating_Temp, °C), yield/tensile strength (MPa) and elongation (%) at temperature, following ASTM E21. When multiple temperatures were available for the same chemistry, measurements were stored as separate rows keyed by Operating_Temp to enable scenario‑conditioned learning rather than collapsing to a single scalar. For summary targets used in optimization, the default label was the tensile strength at the specified application temperature (e.g., 600 °C when available; otherwise the nearest within 550–600 °C). Replicates (n ≥ 3) were averaged (mean) with standard deviations retained in auxiliary fields for uncertainty‑aware analysis. Entries were excluded if gauge‑section temperature control was undocumented or if fewer than three valid replicates were provided.

***\*2.2 Attention‑based Deep Learning Architecture and Training\****

A multitask neural network with a multihead attention module was developed to predict multiple targets while providing interpretability. Standardized numeric inputs were projected to a 128‑dimensional dense embedding. A multihead attention block (8 heads; key/query dimension 64) with residual connection and layer normalization was then applied to capture nonlinear, context‑dependent element–property interactions. The attention operation followed formula, and multihead outputs were concatenated and linearly projected[12-15].

[Figure 1 about here]

where Q represents the query, K the key, V the value, and d_k the key dimension used to scale the dot‑product result. A three‑layer feed‑forward network (256→128→64, ReLU activations, dropout p = 0.2) processed the attention‑enhanced embedding. Task‑specific heads produced the targets under a weighted mean‑squared‑error objective with target‑wise scaling to account for differences in units and variance. Scenario conditioning in this study was limited to a single scenario variable, Operating_Temp, and applied only to high‑temperature tensile targets. Intrinsic targets did not receive any operating inputs.

Models were trained with Adam (initial learning rate 1×10⁻³, β₁=0.9, β₂=0.999, ε=1×10⁻⁸), cosine‑annealing warm restarts, batch size 64, and early stopping (patience=20 epochs). Maximum training length was 500 epochs. Bayesian optimization (TPE) tuned attention heads (4–12), embedding width (64–192), MLP widths (128–384), dropout (0–0.4), and learning rate (3×10⁻⁴–3×10⁻³). L2 weight decay (1×10⁻⁵) and gradient clipping (global‑norm 5.0) were used when beneficial. Evaluation used R², RMSE, and MAPE on the held‑out test set; calibration coverage of 80–90% prediction intervals was assessed via Monte Carlo dropout. Grouped cross‑validation by source/steel category assessed robustness to distribution shift. Attention weights were exported for interpretability (global ranking, property‑specific top‑k, heatmap), which supported Fig. 1. Reproducibility was ensured by fixing random seeds for splitting and initialization and by logging preprocessing parameters, model configurations, and training curves. 

For each head h, given input feature embeddings X ∈ ℝ^{N×d} (N samples, d embedding width), we form query/key/value as follows.


where Q, K, V ∈ ℝ^{d×d_k}. Scaled dot‑product attention is defined as in the Methods text.


Multihead outputs are concatenated and linearly projected: 


In our model, the input rows correspond to the unified vector of composition (wt%) and heat‑treatment parameters for each sample; attention therefore learns context‑dependent weights linking elements/process variables to each property head. We used 8 heads (tuned in {4–12}), head dimension d_k chosen so that total width matches the embedding (e.g., d = 128), and residual + layer normalization to stabilize training. The attention heatmaps reported in the Results section are derived from averaged attention‑induced weights and output attentions, summarized per feature to indicate influence on each property. Uncertainty handling (MC dropout) is applied after attention and MLP layers, so attention‑derived attributions reflect the same stochastic passes used for predictive mean/variance.[16–20]


***\*Fig. 1 Multi-objective optimization of piston steel composition for commercial vehicle based on attention neural network.\****

Loss used target‑wise weighted MSE across tasks; evaluation reported R², RMSE, MAPE on the held‑out test set and an external set. Coverage of nominal 80%/90% prediction intervals was used to check calibration.

***\*2.3\**** ***\*Multi‑objective Optimization (NSGA‑II) and Decision Criteria\****

The inverse design problem was cast as a multi‑objective optimization that seeks to maximize a dimensionless composite performance score while controlling cost and weldability.  The composite score **F** was defined as a weighted sum of three normalized surrogates,

**F = w₁·S + w₂·D + w₃·H (w₁+w₂+w₃=1)**

where **S**, **D** and H represent predicted strength, ductility and high-temperature capability, respectively. Each surrogate was scaled to [0,1] using domain‑specific lower and upper limits; weights were chosen to reflect the relative importance of the three properties in the target application. For inverse design runs aimed at maximizing mechanical performance while minimizing cost, the base objective vector was

**Min F(x)= [−Yield Strength, −Tensile Strength, −Impact Toughness, Cost Index]**

In high‑temperature service scenarios, the high‑temperature tensile strength surrogate replaced or supplemented the room‑temperature tensile strength surrogate with application‑specific weights.  To guard against over‑confident predictions, an uncertainty‑adjusted score **F****rob** was computed for each candidate by subtracting a small multiple of the aggregated predictive standard deviation (obtained via first‑order propagation from per‑target uncertainties) from **F**; rankings by **F** and **F****rob** were compared to ensure robustness, and repeated runs with fixed random seeds confirmed reproducibility.

Decision variables comprised nine alloying elements—carbon (C), chromium (Cr), molybdenum (Mo), vanadium (V), manganese (Mn), silicon (Si), nickel (Ni), phosphorus (P) and sulphur (S)—and three heat‑treatment variables—quench temperature, temper temperature and cooling rate—reflecting controllable levers in industrial practice. Composition bounds were 0.15–0.60 wt% C, 0.50–2.00 wt% Cr, 0.10–0.50 wt% Mo, 0.02–0.30 wt% V, 0.50–1.50 wt% Mn, 0.20–1.20 wt% Si, 0.10–1.50 wt% Ni, P ≤ 0.03 wt% and S ≤ 0.02 wt%.  Process bounds were 820–1050 °C for quench temperature, 150–680 °C for temper temperature and 5–100 °C·min⁻¹ for cooling rate. An aggregate alloying cap of **∑(C+Cr+Mo+V+Mn+Si+Ni)**≤8 wt% enforced overall alloy dilution, while trace‑element violations were clipped. Candidates breaching soft bounds were repaired by proportional renormalization; hard bounds were applied by truncation. 

To reflect weldability and hardenability control, a carbon‑equivalent cap was enforced. We primarily used the IIW carbon equivalent (CEV): **CEV = C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15**. Because Cu was not a decision variable in the base optimization, its value was taken from recorded composition when available; otherwise Cu was set to 0 for feasibility checks. For cases without Cu measurements, we also report a reduced form **CE = C + Mn/6 + (Cr+Mo+V)/5 + Ni/15**, which produced comparable ordering within our composition windows. A hard constraint **CEV ≤ CE_ref** was applied, where CE_ref equals the carbon‑equivalent of the existing piston steel used as the reference. Candidates exceeding the cap were repaired by proportional down‑scaling of the major‑alloy sum followed by re‑clipping to box limits. This constraint was included alongside the aggregate alloy cap and trace‑element limits to ensure manufacturability and joining compatibility.

Cost was computed as a sum of element prices and heat‑treatment energy:

Cost = Σ_i (wt%_i × price_i) + E_Q + E_T.

A relative cost index CI = Cost / Cost_ref normalized costs to the reference piston steel (4140 grade), and a hard constraint CI ≤ 1.30 ensured economic feasibility.

To encourage manufacturability and thermal compatibility, thermal conductivity and thermal expansion were treated as soft constraints with application‑specific reference windows (e.g., κ∈[25,35] W·m⁻¹·K⁻¹ at room temperature; α∈[10,14]×10⁻⁶ K⁻¹). Soft penalties were added to the scalarized ranking used for selection but did not cause candidate rejection: 

Penalty P = λ_κ·max(0, |κ − κ*| − Δκ) + λ_α·max(0, |α − α*| − Δα).

where κ, α denote preferred targets and Δκ, Δα denote acceptable half‑widths. κ and α were predicted by the multi‑task model; when measurements existed for a candidate chemistry, measurements overrode predictions. These soft terms down‑weighted out‑of‑window candidates while preserving Pareto diversity. Unless otherwise specified, the Standard profile was adopted in this study: κ = 28 W·m⁻¹·K⁻¹, Δκ = 5 W·m⁻¹·K⁻¹, λ_κ = 0.8; α = 12×10⁻⁶ K⁻¹, Δα = 3×10⁻⁶ K⁻¹, λ_α = 0.8.

NSGA‑II used a population of 200, 500 generations, Simulated Binary Crossover (probability 0.9; distribution index η_c=20), polynomial mutation (probability 0.1; η_m=20), tournament selection (size=2), and elitist non‑dominated sorting with crowding distance. Initialization sampled within bounds with trace‑element clipping and a soft cap on the sum of major alloying elements. Convergence was monitored by hypervolume improvement and Pareto set stability; diversity was promoted via crowding and Euclidean dispersion in composition subspace. Final selection targeted three decision zones (high‑performance, balanced, cost‑sensitive) and ensured representation of compositional styles to avoid redundancy. Selection criteria required Pareto rank 1, predicted gains ≥20% in key metrics, cost increase ≤30%, manufacturability and process compatibility, and adequate separation in composition space.Sensitivity analyses perturbed process parameters (±10~20°C for temperatures; ±10% for cooling rate) and, for scenario‑dependent targets, operating variables (Operating_Temp ±25~50°C; frequency ±10%) to assess robustness.

***\*2.4 E\*******\*xperiment Methods\****

Experimental validation was conducted to verify model predictions and to establish manufacturability under industrially relevant windows. Two Pareto‑optimal candidates were down‑selected: a quenched‑and‑tempered (QT) steel emphasizing strength–toughness balance and elevated‑temperature performance, and a non‑quenched‑and‑tempered (NQT) steel emphasizing simplified processing and cost control.

Heats of 50 kg were produced by vacuum induction melting (VIM) under argon. Electrolytic iron (99.9% purity) and master alloys (FeCr, FeMo, FeV) were used as charge materials. Melting was performed at ~1650 °C with a 30 min homogenization hold. Ingots were hot forged at 1200 °C with a 4:1 reduction ratio and controlled‑cooled to ~500 °C to minimize thermal gradients. Representative equipment included a VIM system and a hydraulic forging press.

Specimens were austenitized at 860 °C for 30 min and oil‑quenched to room temperature. Tempering was conducted for 2 h with air cooling (QT‑Steel at 580 °C; NQT‑Steel at 200 °C for stress‑relief when applicable). Furnace temperature uniformity was verified by a type‑K thermocouple array. Industrial windows were respected to ensure scalability.

EBSD mapping was performed on a field‑emission SEM (FEI Quanta 450 FEG‑SEM, Thermo Fisher Scientific, USA) equipped with an EBSD detector (Oxford Symmetry, Oxford Instruments, UK). Samples were mechanically polished to 0.25 μm and electropolished in 10% perchloric acid/ethanol solution (perchloric acid, ≥70%, Sigma‑Aldrich, USA). Acquisition settings were 20 kV accelerating voltage, 70° specimen tilt, and 0.5 μm step size. Data were processed in AZtec (Oxford Instruments) with zero‑solution cleanup and standard grain size statistics. TEM was carried out on a FEI Tecnai G2 F20 (Thermo Fisher Scientific, USA) at 200 kV. Thin foils were prepared by twin‑jet electropolishing in 5% perchloric acid/ethanol at −30°C. Bright‑field imaging and SAED were performed for phase identification and orientation relationships; precipitate size distributions and number densities were quantified using ImageJ with calibrated pixel size. 

Tensile tests were performed according to ASTM E8/E8M using sub‑size dog‑bone specimens (gauge length 25 mm) on a universal testing machine (Instron 5982, Instron, USA) at room temperature with a strain rate of 1×10⁻³ s⁻¹ (n=5 per composition). Strain was measured using a clip‑on extensometer (gauge length 12.5 mm). Charpy V‑notch impact testing followed ASTM E23 using 10×10×55 mm specimens on a 300 J pendulum (n=10, room temperature). High‑temperature tensile tests followed ASTM E21 at selected temperatures (e.g., 500~600°C); yield/tensile strength and elongation at temperature were recorded (n≥3 per condition). 

 

***\*3 Results\****

***\*3.1 Database overview\**** 

In constructing the dataset, we first collated more than 1000 published steel formulations and categorized the samples as low-alloy, medium-alloy, and high-alloy steels according to the total content of alloying elements, and then compared them to the total content of alloying elements, the content of main alloying elements in low alloy steel is less than 5% , that in medium alloy steel is between 5% and 12% , and that in high alloy steel is more than 12%. Fig. 2(a) shows the distribution of samples under this classification: Low-alloy and carbon steels are the most abundant, while high-alloy steels account for only 5%. Fig. 2(b) shows the Pearson correlation matrix for the major alloying elements, reflecting the pattern of element combinations in the formulation. For example, elements such as Cr, Ni, Mo and V, which are commonly used to improve strength and hardness, are significantly positively correlated with each other, while carbon is negatively correlated with them. In addition, Al is also negatively correlated with Ni and Cu, it is suggested that other elements should be considered in the control of aluminum content. To understand the distribution and interrelationships of the performance data, we plotted several scatter plots. Fig. 2(c) depicts the distribution of yield strength and impact toughness over a wide range of samples, indicating that they may be in different equilibriums for different formulation combinations and are influenced by the content of a particular element. Fig. 2(d) shows a marked decrease in the thermal conductivity of the alloy as the total content of strong metallic elements such as Cr, Ni and Mo increases; this trend reflects the scattering of electrons and phonons by heavy alloying elements. Fig. 2(e) shows the dependence of the high-temperature strength on Mo content, which is effective at increasing the high-temperature strength, but the gain decreases above about 2 wt%. Finally, Fig. 2(f) shows the relationship between the comprehensive performance index and the cost index. The high dispersion of points indicates that there are multiple cost-performance combinations in the sample, implying that cost is not the only determinant of performance; This is a trade-off that needs to be considered in subsequent design optimizations.

[Figure 2 about here]

***\*Fig. 2 Schematic of database analysis: (a) sample classes; (b) element correlations; (c) yield vs. impact toughness; (d) (Cr+Ni+Mo) vs. thermal conductivity; (e) Mo vs. high‑temperature strength; (f) composite performance vs. cost index.\****

***\*3.2 Model benchmarking and method comparison\****

For the regression prediction of steel properties, this paper constructs four types of models: traditional regression, classical machine learning models (random forest/gradient boosting, etc.), deep neural networks, and deep models based on attention mechanism. The coefficient of determination R² was used as the precision index, and the training time (minutes) was recorded to compare the model efficiency[6,7]. The results are shown in Fig. 3(a). The traditional regression model R² is 0.72, and the calculation time is the shortest; the classical machine learning model increases R² to 0.85; the deep learning model further increases R² to 0.89; the deep learning model with attention mechanism can not only capture the non-local association between features, but also highlight the key information. The prediction accuracy is improved to 0.94, but the training time is also increased to about 20 min. The attention module can select the key areas in the input and filter the noise gradient, enhance the ability of feature representation, and provide in-depth interpretation of network decision-making.

In order to explore the optimal formulations while satisfying the constraints of strength, toughness, thermal conductivity, cost, etc., we compare several mainstream multi-objective optimization algorithms: genetic algorithm (GA), particle swarm optimization (PSO), fast non-dominated sorting genetic algorithm (NSGA-II), multi-objective particle swarm optimization (MOPSO) and the proposed method. GA simulates natural selection, using selection, crossover and mutation to iterate and evolve among populations[11,14]. With reliable global search ability, PSO allows particles to constantly adjust their speed and position according to their historical best and neighborhood best. NSGA-II introduces fast non-dominated sorting and crowding distance evaluation on the basis of GA framework, then a Pareto front solution set is obtained. Fig. 3(b) shows the relationship between the convergence algebra of each algorithm and the quality of the solution set: GA and PSO converge slowly and the quality of the final solution is low; NSGA-II and MOPSO improve the quality of the solution set; The proposed method fuses the high-precision attention surrogate model, adaptive penalty, and diversity maintenance, consistently obtains the highest solution set quality and converges in advance throughout the iteration process, and improves the performance of the algorithm, showing that accurate surrogate models and targeted optimization strategies are essential for the design of complex alloys.

To comprehensively evaluate the performance of different methods, we constructed radar charts from five dimensions of interpretability, robustness, efficiency, accuracy, and generalization ability, as shown in Fig. 3(c). Traditional machine learning models have strong interpretability but limited accuracy and generalization ability; deep learning models have high accuracy but poor interpretability. After introducing the attention mechanism, the model not only significantly improves the prediction accuracy, but also reflects the importance of each input feature through the attention weight, thus enhancing interpretability. In the multi-objective optimization algorithm, GA and PSO are simple but inefficient and unstable; NSGA-II and MOPSO are better at balancing multiple goals; The "attention + multi-objective optimization" framework proposed in this paper achieved the highest scores on all five indicators.

[Figure 3 about here]

***\*Fig. 3 Model and optimizer comparison: (a) prediction models; (b) optimization algorithms; (c) radar chart of five criteria.\****

***\*3.3 Predictive performance and interpretability of the attention‑based model\****

The predictive performance on held‑out test sets was evaluated for all target properties using parity analysis, error summary statistics, calibration coverage, and grouped cross‑validation. Parity plots displayed the distribution of predictions versus measurements with 1:1 reference lines and ±10% error bands (Fig. 4). Each sub-graph corresponds to six key performance indicators: (a) tensile strength, (b) yield strength, (c) elongation, (d) impact toughness, (e) thermal conductivity and (f) coefficient of thermal expansion. The blue dots are the actual versus predicted values of the test sample, the red lines are linear fits with a slope close to 1, and the shaded areas represent 95% confidence intervals. Coefficient of determination (R²) values are all above 0.95 (the highest is 0.982), which shows that the model has a high degree of prediction and fitting for all kinds of properties. The confidence interval is narrow, indicating small prediction error and good generalization. The attention mechanism is the key to this performance improvement and provides interpretability by visualizing attention weights.

[Figure 4 about here]

***\*Fig. 4 Prediction model performance validation: attention‑based model predictions on the test set vs. measurements for (a) tensile strength; (b) yield strength; (c) elongation; (d) impact toughness; (e) thermal conductivity; (f) coefficient of thermal expansion. Points: experimental data; red line: linear fit with 95% confidence band.\****

In order to further clarify the decision-making basis of the surrogate model, we conduct a visual analysis of the attention weight. Fig. 5(a) shows an attention-weighted heat map, with rows representing the input characteristics (chemical element content and heat treatment parameters) and columns representing the performance indicators predicted by the model. The more red the color is, the more attention the model pays to the feature when predicting the corresponding performance, and the bluer the color is, the lower the weight is. For example, carbon (C) has a higher weight in tensile strength, yield strength, thermal conductivity and density, while chromium (Cr) and molybdenum (Mo) are significantly red in high-temperature strength; some process parameters such as quenching temperature, tempering temperature, and cooling rate have greater influence weights on hardness and strength, which is consistent with traditional metallurgical knowledge. Fig. 5(b) averages the attention weights for all the performances and ranks them by size to get the overall feature importance. Fig. 5(c) shows the top features with the highest attentional weight for each metric.

Attention weight analysis not only makes the model have high predictive ability, but also explains the decision-making basis of the model by explicitly displaying the feature contribution. The attention mechanism can automatically select key areas in the input and suppress noise, enhancing feature representation and interpretability. These analysis results also verify that the laws captured by the model are consistent with metallurgical knowledge, follow-up formula optimization provides a credible scientific basis.

[Figure 5 about here]

***\*Fig. 5 Attention Weight Analysis: (a) attention weight heatmap of input features for each performance indicator; (b) average attention weight ranking; (c) key feature bar graphs for each performance indicator.\****

***\*3.4 Multi‑objective optimization outcomes (NSGA‑II)\****

We developed a framework that couples an interpretable, attention‑based surrogate model with NSGA‑II to design piston steel compositions under weldability and cost constraints. The optimization explored nine alloying elements and three heat‑treatment parameters and generated a diverse population of feasible chemistries. Fig. 6(a) shows that the resulting non‑dominated set forms a Pareto frontier (red points) on the cost–performance plane, with a clear knee region around a cost index of 300–320 (i.e., ~20–30% higher than the reference steel) where overall performance scores exceed 0.96; solutions beyond the dashed line were rejected due to exceeding the +30% cost cap. As expected, strength and ductility are inversely correlated, yet the NSGA‑II search identifies a narrow band of high‑fitness designs that balance these properties (Fig. 6(b)). Convergence diagnostics (Fig. 6(c)) show that both the best and average fitness scores rise sharply in the first 30–40 generations and stabilise thereafter, with the solution range steadily narrowing—evidence of rapid convergence and maintained diversity.

[Figure 6 about here]

***\*Fig. 6 Multi‑objective optimization results: (a) performance–cost Pareto distribution with cost cap; (b) strength–ductility trade‑off coloured by CEV; (c) convergence of best/average fitness.\****

To down‑select implementable alloys, we applied a stepwise filtering process (Fig. 7(a)). Starting from ~2000 candidate solutions, imposing the cost cap reduced the population by roughly 25 %, and applying the carbon‑equivalent limit (CEV ≤ 0.60) halved it again. Restricting to the Pareto set further narrowed the field to a few dozen chemistries, and final screening based on manufacturability, process feasibility and composition diversity yielded only a handful of steels for experimental validation. The inset hypervolume plot confirms monotonic improvement of the non‑dominated set, reinforcing the reliability of the optimisation scheme. When benchmarked against conventional quench‑tempered and non‑quench‑tempered grades, the AI‑designed alloys offer markedly superior composite performance at comparable or modestly higher costs (Fig. 7(b)): the AI‑designed non‑quenched steel (Ai‑NQT) achieves the highest performance (~0.72) at a cost index near 40, whereas the AI‑designed quenched steel (Ai‑QT) delivers a balanced performance (~0.50) within the cost‑feasible zone. In contrast, conventional steels cluster at lower performance levels and either low or moderate costs. These trends illustrate that the optimisation simultaneously unearths high‑performance options and cost‑sensitive trade‑offs suited to different service requirements.

[Figure 7 about here]

***\*Fig. 7 Screening flow and candidate comparison: (a) screening flow: starting from all solutions, cost constraint, weldability constraint and Pareto set are applied successively, and the remaining quantity of each stage is shown; (b) comparison of alternatives: bubble plot of comprehensive performance versus cost index comparing AI‑designed steel with conventional steel, colour indicates carbon equivalent.\****

***\*3.5 Microstructure and mechanical properties\****

Fig. 8 shows the microstructures of AI‑designed steels in quenched‑and‑tempered (QT) and non‑quenched (NQT) conditions. The upper row (a1–a3) corresponds to the AI‑designed QT steel. In the back‑scattered electron image (a1) the matrix comprises fine pearlite (P) and ferrite (F); pearlite lamellae appear slender and uniformly distributed between ferritic grains. The EBSD orientation map (a2) shows aligned lamellae, consistent with a refined, lamellar microstructure. The phase map (a3) reveals a BCC matrix with a minor FCC fraction at lamellar boundaries (retained austenite). The lower row (b1–b3) depicts the AI‑designed NQT steel with a more uniform ferrite–pearlite mixture and equiaxed grains; the phase map is entirely BCC.
[Figure 8 about here]

***\*Fig. 8 Analysis of microstructure: (a1)(b1) OM; (a2)(b2) IPF; (a3)(b3) Phase\****

Fig. 9 summarises the mechanical performance of AI‑designed and conventional steels. In the quenched‑and‑tempered condition (Fig. 9(a)), the AI‑designed QT steel displays a yield strength of approximately 1000 MPa and a tensile strength of about 1300 MPa. These values are notably higher than those of the conventional QT steel (around 880 MPa yield and 990 MPa tensile strength). The strength advantage arises from the refined pearlite–ferrite lamellae and the presence of retained austenite in AI‑QT steel, which impede dislocation motion and provide transformation strengthening. The AI‑NQT steel, which omits quenching, still achieves a yield strength of roughly 780 MPa and a tensile strength near 950 MPa, exceeding the conventional NQT steel (600 MPa yield, 860 MPa tensile). Its strength improvement is attributed to controlled alloying and fine ferrite grain size achieved via the AI‑guided composition design. Fig. 9b compares ductility (elongation) and impact toughness among the four steels. The AI‑QT steel exhibits an elongation of ~16 % and an impact toughness of ~27 J. In contrast, the AI‑NQT steel combines moderate strength with superior toughness: its elongation (~18 %) surpasses that of the AI‑QT steel and its impact toughness (~39 J) is the highest among all steels, indicating excellent resistance to crack propagation. The conventional steels have greater elongation (up to ~21 %) but lower strength; their impact toughness values (28~31 J) are lower than that of the AI‑NQT steel. These results show that the AI‑designed alloys achieve a favourable balance between strength and ductility/toughness compared with conventional grades.

[Figure 9 about here]

***\*Fig. 9 Mechanical properties: (a) strength; (b) ductility and impact toughness.\****

***\*3.6 Physical properties and heat resistance\****

Good thermal properties and high temperature oxidation resistance are necessary conditions for piston steel to serve in severe combustion environment. By comparing the temperature-dependent thermal diffusivity, specific heat capacity, and thermal conductivity, as well as the mass gain, oxide film thickness, and cross-sectional morphology of the long-time oxidation test, the present section provides a more comprehensive understanding of the mechanism of the oxidation process, evaluation of thermal management capabilities and heat resistance of AI-optimized and conventional steels.

Fig. 10(a–c) shows the variations of thermal diffusivity α, specific heat capacity Cp and thermal conductivity λ for the four tested steels in the range from room temperature to about 900 °C. With the increase of temperature, the thermal diffusivity of all steels decreases, which is due to the hindrance of thermal diffusion caused by thermal vibration enhancement. Compared with conventional steel, the thermal diffusivity of AI‑optimized steel is significantly lower in the range of 20–200 °C, which may be attributed to its multi‑alloying and fine precipitated phases increasing lattice scattering; however, around 700–850 °C, the thermal diffusivity of AI‑optimized steel is significantly lower than that of conventional steel, all curves have trough values, corresponding to the α → γ phase transformation, and the difference between the steels decreases at this stage, indicating that the thermal diffusion properties near the high temperature phase transformation point are less affected by the alloy type.

The specific heat capacity increases with the increase of temperature, and there is an obvious inflection point at 700–800 °C, which is caused by the endothermic effect of austenitizing. The Cp curves of AI‑optimized steel are slightly lower than those of traditional steel at high temperature, which means that the heat storage capacity of AI‑optimized steel is equivalent to that of the control material and will not adversely affect the thermal fatigue performance.

The trend of thermal conductivity is similar to that of thermal diffusivity, which decreases with the increase of temperature, and a trough is formed near the γ phase region. The thermal conductivity of AI‑optimized steel is slightly lower from room temperature to 400 °C, which is related to its higher alloying element content and the enhancement of phonon scattering caused by the microstructure; by 800 °C, the thermal conductivity of AI‑optimized steel is slightly lower than that of AI‑optimized steel, and the thermal conductivity of each steel converges to 30–35 W·m⁻¹·K⁻¹, which meets the requirements of heat conduction when the piston works. Overall, the AI‑designed steel maintains the same or even better thermal diffusion and conduction ability as the traditional steel while ensuring the improvement of mechanical properties.

[Figure 10 about here]

***\*Fig. 10 Physical properties: (a) thermal diffusivity; (b) heat capacity; (c) thermal conductivity.\****

The new steel is designed to serve the next generation of heavy‑duty diesel engines, and its service temperature has been raised to 600 °C, so we compared the isothermal oxidation behavior of four steels at 600 °C. Fig. 11 summarises the oxidation kinetics of the four steels at 600 °C. The mass‑gain curves (Fig. 11a) show roughly parabolic behaviour for all alloys, consistent with diffusion‑controlled oxidation, but the conventional Q&T steel oxidises fastest, reaching > 1.2 mg·cm⁻² after 500 h. In contrast, the AI‑optimised Q&T steel gains only ~0.8 mg·cm⁻² over the same period. Similar trends are seen for the non‑quenched steels. Thickness measurements of the oxide layers (Fig. 11b) further illustrate the advantage of the AI‑optimised alloys: after 500 h the oxide scales on conventional Q&T and NQT steels are approximately 116 μm and 89 μm, respectively, whereas those on the AI‑QT and AI‑NQT steels are only around 62 μm and 54 μm. Thus, the AI‑designed alloys oxidise more slowly and form thinner scales.

Cross‑sectional SEM images (Fig. 11c) reveal that the oxide layers on conventional steels are rough, porous and poorly adherent, whereas those on the AI‑optimised steels are dense, uniform and well bonded to the substrate. This improvement derives from the tailored combination of alloying elements: higher Cr and Si levels favour the rapid formation of protective Cr₂O₃ and SiO₂‑rich films that block oxygen diffusion, while minor additions of Mo and V enhance scale adhesion and stability.

[Figure 11 about here]

***\*Fig. 11 Oxidation behavior: (a) mass gain; (b) oxide layer thickness; (c) cross‑sections at 600 °C.\****

 

***\*4 Discussion\****

***\*4.1\**** ***\*Interpreting principal findings in the context of alloy design and prior literature\****

This study presents an interpretable multi‑task surrogate model and a constrained NSGA‑II optimizer to design piston steel compositions. The surrogate achieved calibrated prediction accuracies on held‑out tests (R² ≈ 0.89–0.94 for single‑task regressions; ≈0.95–0.98 for the integrated attention‑based model), with ±10% error coverage up to 92%, across strength, ductility, high‑temperature tensile properties and oxidation resistance. Attention weights identified carbon, chromium and molybdenum contents and quench/temper processing as dominant drivers of strength and toughness, corroborating metallurgical expectations. Under weldability and cost constraints, the NSGA‑II search produced 36 non‑dominated solutions, mapping out high‑performance, balanced and cost‑sensitive regions of the design space. Compared with the reference 4140 steel, representative AI‑designed alloys delivered 18–25% higher yield and tensile strengths, modest gains in elongation, nearly doubled impact toughness and improved thermal stability, all at cost increases ≤30%. High‑temperature diffusivity and conductivity curves showed that AI‑designed alloys retained adequate heat‑transfer capacity while delaying α→γ phase transitions. Oxidation tests at 600 °C for 500 h revealed thinner, denser oxide scales and lower mass gain on AI‑designed steels than on conventional steels, indicating superior scale‑forming ability without Al additions.

These findings address the three research questions posed in the introduction: (i) whether a calibrated, interpretable model can serve as a surrogate for steel design; (ii) whether a constrained multi‑objective search yields a structured Pareto front; and (iii) whether down‑selected compositions can be manufactured and validated. The affirmative answers advance data‑driven alloy design by integrating interpretability with decision support. Prior work often applied single‑objective search or black‑box models that offered limited insight into chemical levers. Here, the attention mechanism makes feature–property relationships transparent, while the NSGA‑II front elucidates performance–cost trade‑offs and highlights multiple viable compositional routes, including Cr‑dominated, Mo‑enhanced, V‑strengthened, Ni‑tuned and Si‑hardened pathways.

***\*4.2\**** ***\*Trade‑offs, design‑space structure, and implications for manufacturability\****

The constrained Pareto set reveals inherent trade‑offs among mechanical properties, thermal performance and cost. Strength and ductility are negatively correlated: achieving a yield strength around ≈1000 MPa via high‑C, high‑Cr/Mo chemistries and rapid quenching yields elongations of ≈15%, whereas reducing alloy content and tempering more gently lowers strength to ≈780 MPa but increases elongation and impact toughness. The composite performance versus cost plot exhibits a “knee” near a cost index of 1.2–1.3 (20–30% above the reference), beyond which further alloying yields diminishing returns. This knee is an attractive operating point for industry, balancing elevated performance against alloy surcharge.

The NSGA‑II optimizer employs fast non‑dominated sorting with elitism and crowding distance to maintain diverse solutions while reducing computational complexity. Hypervolume monitoring confirmed convergence and diversity, with the Pareto set stabilizing after ≈420 generations. By exploring nine alloying elements (C, Cr, Mo, V, Mn, Si, Ni, P, S) and three heat‑treatment variables, the search uncovered multiple “chemical styles” that satisfy both weldability and cost caps, allowing manufacturers to select candidates based on alloying element availability or pricing volatility. Hard constraints on carbon equivalent (CEV ≤ 0.60) ensured weldability, while soft penalties on thermal conductivity and expansion confined solutions to application‑specific windows.

Manufacturability is further supported by compositional and process bounds grounded in industrial practice. All solutions respect typical ranges for quench/temper temperatures (820–1050 °C) and temper temperatures (150–680 °C) and cooling rates (5–100 °C/min), as well as aggregate alloy caps and trace‑element limits. The two AI‑designed steels subjected to experimental validation—AI‑QT and AI‑NQT—were processed using conventional VIM casting, hot forging, and controlled heat treatment. Their success demonstrates that the proposed design framework yields alloys amenable to existing production lines and welding operations. Moreover, sensitivity analyses indicated that modest perturbations in processing conditions (±10–20 °C) or operating temperatures (±25–50 °C) did not reorder the Pareto solutions, implying robust manufacturability under variable factory conditions.

***\*4.3 Transmission electron microscopy analysis\****

In the AI‑QT steel (Fig. 12(a1)), bright‑field imaging reveals well‑aligned lamellar pearlite colonies with alternating ferrite and cementite lamellae. The lamellae are very fine—high‑resolution TEM in Fig. 12(a2) shows that the spacing between cementite plates is on the order of 10 nm, substantially smaller than in conventional steels. Selected‑area diffraction from this region confirms the presence of α‑ferrite (BCC) and θ‑cementite (orthorhombic) phases. Such ultrafine lamellae impede dislocation motion and contribute to the high yield and tensile strengths observed for AI‑QT steel. Moreover, Fast Fourier Transform (FFT) patterns taken from three regions labelled A, B and C in Fig. 12(a2) exhibit diffuse spots consistent with nanoscale retained austenite films at ferrite/pearlite boundaries. These thin films can transform to martensite under stress, providing a transformation‑induced plasticity (TRIP) effect that improves impact toughness without sacrificing strength.

The AI‑NQT steel displays a different morphology. As shown in Fig. 12(b1), the matrix consists predominantly of equiaxed ferrite grains with sparse pearlite or bainitic ferrite. The reduced lamellar content and larger ferrite grains explain why AI‑NQT steel has lower strength than AI‑QT steel but superior elongation and toughness: the absence of continuous cementite lamellae reduces stress concentration and crack initiation sites.

Higher‑resolution scanning TEM of the AI‑NQT steel (Fig. 12(b2)) reveals a high density of nanoscale precipitates. High‑angle annular dark‑field (HAADF) imaging shows bright particles within the ferrite matrix, while the accompanying energy‑dispersive X‑ray spectroscopy (EDS) maps demonstrate that these precipitates are enriched in manganese and sulphur (purple and blue signals) as well as vanadium and carbon (yellow and red signals). This indicates that complex (Mn,S) inclusions decorated with vanadium carbides (VC) have formed. Such precipitates act as nucleation sites for secondary carbides and help refine ferrite grains.

Atomic‑resolution images (Fig. 12(b2)) from the highlighted precipitates further clarify their structure. Lattice imaging reveals coherent or semi‑coherent interfaces between MnS and VC, with measured interplanar spacings of ~0.15 nm and ~0.35 nm and misorientation angles around 53°, 71°, 142° and 6°. FFT patterns associated with these regions support the orientation relationships. The close crystallographic match between MnS and VC suggests that vanadium carbides nucleate epitaxially on MnS inclusions, producing composite particles that are more effective in pinning dislocations than either phase alone. This microstructural feature explains why the AI‑NQT alloy achieves a favourable strength–toughness balance despite the absence of TRIP‑induced austenite.

 

 

Taken together, the TEM analysis shows that the AI‑optimised compositions promote two distinct strengthening mechanisms: ultrafine lamellar pearlite with retained austenite films in the quenched‑and‑tempered alloy and a high density of coherent MnS + VC precipitates in the non‑quenched alloy. Both microstructures are consistent with the alloying pathways identified by the optimisation (Cr‑dominated and V‑strengthened routes, respectively). These nanoscale observations therefore validate the model‑guided design and provide mechanistic insight into how the chosen alloying elements and heat treatments lead to the observed macroscopic mechanical properties.

[Figure 12 about here]

***\*Fig. 12 TEM analysis: (a1) Bright‑field TEM of AI‑QT steel showing refined lamellar pearlite (inset: SAED); (a2) high‑resolution TEM and FFT from regions A–C indicating nanoscale retained austenite films; (b1) bright‑field image of AI‑NQT ferrite matrix with sparse pearlite/bainitic ferrite; (b2) HAADF‑STEM and EDS maps revealing MnS + VC composite precipitates.\****

***\*5\*******\*. Conclusion\****

This study developed an interpretable, attention-based deep learning framework integrated with NSGA-II to design commercial vehicle piston steels under industrially realistic constraints. By separating intrinsic descriptors from operating variables, the model achieved calibrated accuracy across strength, ductility, high-temperature performance, and oxidation resistance, while attention weights provided property-specific interpretability consistent with metallurgical knowledge.

The constrained multi-objective search generated a structured Pareto front with 36 non-dominated solutions spanning high-performance, balanced, and cost-sensitive regions. Experimental validation confirmed that the AI-designed steels met weldability and cost caps, while achieving 18–25% performance gains, refined microstructures, and significantly improved oxidation resistance at 600 °C/500 h. The comprehensive performance radar chart further highlights that the AI-optimized steels consistently outperform conventional grades across strength, toughness, thermal stability, and cost dimensions.

[Figure 13 about here]

Limitations remain in data heterogeneity, sparse microstructural descriptors, and reliance on surrogate predictions. Future work should emphasize standardized high-temperature datasets, physics-aware conditioning, microstructure-informed features, and robust uncertainty quantification.

Overall, the proposed framework demonstrates that calibrated interpretability coupled with constrained optimization provides a practical and generalizable pathway from data to deployable piston steel compositions, with broader applicability to multi-objective alloy design.
 
***\*Data availability\****
 
The data that support the findings of this study are available from the corresponding author upon reasonable request.
 
***\*CRediT author statement\****
 
Weitao Ma: Methodology, Investigation, Writing – original draft. Yanjun Rao: Data curation, Validation. Zheyue Zhang: Software, Visualization. Zijian Zhang: Formal analysis. Shuai Zhao: Resources. Renbo Song: Conceptualization, Supervision, Funding acquisition, Writing – review & editing. Guanwen Dai: Resources, Project administration. Yongjin Wang: Investigation. Zengjian Feng: Validation.

***\*Declaration of Competing Interest\****

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

***\*Acknowledgments\****

The authors are grateful for the support from the National Natural Science Foundation of China (No. 52074033) and the Hebei Steel Group Key Research and Development Project (HG2023242).

***\*Figure captions\****

Fig. 1 Multi‑objective optimization of piston steel composition for commercial vehicle based on attention neural network.
Fig. 2 Schematic of database analysis: (a) sample classes; (b) element correlations; (c) yield vs. impact toughness; (d) (Cr+Ni+Mo) vs. thermal conductivity; (e) Mo vs. high‑temperature strength; (f) composite performance vs. cost index.
Fig. 3 Model and optimizer comparison: (a) prediction models; (b) optimization algorithms; (c) radar chart of five criteria.
Fig. 4 Prediction model performance validation: attention‑based predictions vs. measurements for six properties; red line: linear fit; band: 95%.
Fig. 5 Attention weight analysis: (a) heatmap of feature importance per property; (b) averaged ranking; (c) top features per metric.
Fig. 6 Multi‑objective optimization: (a) performance–cost Pareto with cap; (b) strength–ductility trade‑off (CEV colour); (c) convergence.
Fig. 7 Screening flow and candidate comparison.
Fig. 8 Microstructures of AI‑designed steels in QT and NQT conditions with OM/IPF/Phase maps.
Fig. 9 Mechanical properties: strength, ductility and impact toughness.
Fig. 10 Physical properties: thermal diffusivity, heat capacity, thermal conductivity.
Fig. 11 Oxidation behaviour at 600 °C.
Fig. 12 TEM analysis of AI‑QT and AI‑NQT steels.
Fig. 13 Comprehensive performance radar chart.

***\*Reference\*******\*s\****

[1] Hu B, Shi C, Zhang Z, Liu CT. Design of advanced high strength steels by new metallurgical principles. Sci. China Mater. 64 (2021) 1639–1660. https://doi.org/10.1007/s40843-020-1482-9

[2] Li Y, et al. A high-strength and ductile medium-entropy alloy. Nat. Commun. 10 (2019) 4063. https://doi.org/10.1038/s41467-019-11964-1

[3] Xu J, et al. Development of next-generation bainitic steels with improved strength–toughness balance. Mater. Sci. Eng. A 856 (2022) 143977. https://doi.org/10.1016/j.msea.2022.143977

[4] Zhang H, et al. Alloy design strategies for high-performance structural steels: recent advances and future trends. Acta Mater. 252 (2023) 118982. https://doi.org/10.1016/j.actamat.2023.118982

[5] Wang Y, et al. Microstructure and properties of high-performance steels designed by data-driven approaches. J. Alloys Compd. 830 (2020) 154645. https://doi.org/10.1016/j.jallcom.2020.154645

[6] Yang Z, et al. Machine-learning-accelerated design of high-performance steels. npj Comput. Mater. 7 (2021) 84. https://doi.org/10.1038/s41524-021-00543-5

[7] Liu Y, et al. Multi-objective optimization design of alloy steels using machine learning and genetic algorithms. Mater. Des. 223 (2022) 111151. https://doi.org/10.1016/j.matdes.2022.111151

[8] Zhang Y, Li X, Xue D. Machine learning for design and discovery of materials. J. Materiomics 6 (2020) 397–411. https://doi.org/10.1016/j.jmat.2019.12.001

[9] Chen C, Ye W, Zuo Y, Zheng C, Ong SP. Graph networks as a universal machine learning framework for molecules and crystals. Chem. Mater. 33 (2021) 7489–7515. https://doi.org/10.1021/acs.chemmater.0c04473

[10] Wang AY‑T, et al. Machine learning for materials scientists: An introductory guide toward best practices. Chem. Mater. 32 (2020) 4954–4965. https://doi.org/10.1021/acs.chemmater.0c01907

[11] Xiong J, et al. Data-driven discovery of high-strength steels via machine learning. Acta Mater. 196 (2020) 340–350. https://doi.org/10.1016/j.actamat.2020.06.044

[12] Xie T, Grossman JC. Hierarchical graph representation of crystal structures using neural networks. Nat. Commun. 10 (2019) 1–10. https://doi.org/10.1038/s41467-019-12809-7

[13] Lu S, et al. Accelerated design of steel compositions via machine learning and high-throughput experiments. Acta Mater. 219 (2021) 117262. https://doi.org/10.1016/j.actamat.2021.117262

[14] Sun H, et al. Deep learning in materials science: recent progress and emerging applications. Adv. Sci. 10 (2023) 2204891. https://doi.org/10.1002/advs.202204891

[15] Zhang R, et al. Interpretable machine learning for alloy design: challenges and opportunities. npj Comput. Mater. 10 (2024) 22. https://doi.org/10.1038/s41524-024-01118-3

[16] Deb K, Jain H. An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, Part II: handling constraints and uncertainties. IEEE Trans. Evol. Comput. 23 (2019) 210–224. https://doi.org/10.1109/TEVC.2018.2871038

[17] Ishibuchi H, et al. Evolutionary multi-objective optimization: recent developments and future directions. IEEE Comput. Intell. Mag. 15 (2020) 14–28. https://doi.org/10.1109/MCI.2019.2950648

[18] Li K, et al. Efficient multi-objective optimization for materials design with surrogate models. Mater. Des. 182 (2019) 108014. https://doi.org/10.1016/j.matdes.2019.108014

[19] Zhang T, et al. Hybrid multi-objective evolutionary algorithms for materials design optimization. Comput. Mater. Sci. 193 (2021) 110360. https://doi.org/10.1016/j.commatsci.2021.110360

[20] He J, et al. Multi-objective optimization of advanced steels using surrogate models and evolutionary algorithms. Mater. Sci. Eng. A 852 (2022) 143650. https://doi.org/10.1016/j.msea.2022.143650

[21] Wang S, et al. Recent progress in evolutionary algorithms for materials informatics. J. Mater. Inform. 3 (2023) 2023001. https://doi.org/10.20517/jmi.2023.01

[22] Zhao Y, et al. Robust design of alloy compositions via uncertainty-aware multi-objective optimization. npj Comput. Mater. 7 (2021) 103. https://doi.org/10.1038/s41524-021-00548-0

[23] Chen Y, et al. Integrating machine learning and multi-objective optimization for accelerated materials design. Mater. Today 54 (2022) 10–21. https://doi.org/10.1016/j.mattod.2022.01.010

[24] Gao P, et al. Data-driven multi-objective optimization of alloy systems under processing constraints. Comput. Mater. Sci. 184 (2020) 109917. https://doi.org/10.1016/j.commatsci.2020.109917

[25] Xu Q, et al. Advances in surrogate-assisted evolutionary algorithms for materials design. Prog. Mater. Sci. 140 (2024) 101056. https://doi.org/10.1016/j.pmatsci.2023.101056

 