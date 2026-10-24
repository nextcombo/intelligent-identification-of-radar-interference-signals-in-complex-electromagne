# Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Types of Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]

Complex Electromagnetic Environments: Intelligent Identification of Radar Interference Signals Based on Nine Types of Interference Modeling and Multi-Domain Feature Analysis
Keywords: Radar jamming identification; Jamming modeling; Time-frequency analysis; Feature extraction; Decision tree; Convolutional neural network; Small sample learning
1 Introduction
1.2 Overview of existing methods for a long time, the research on interference identification mainly along two technical routes:
1. A shallow classification approach based on artificial features. This method manually designs discriminative quantities with clear physical meanings in time domain, frequency domain, and time-frequency domain, such as signal energy, bandwidth, pulse width, spectral symmetry, sparsity, etc., and then utilizes shallow classifiers like Support Vector Machines, Decision Trees, and Random Forests to complete the decision-making. Its advantages include strong interpretability of features, low computational overhead, and minimal requirement for training samples, achieving near practical recognition accuracy under specific scenarios; however, its shortcomings lie in the highly dependent nature of feature design on expert experience, which limits generalization capabilities when facing novel or compound interference.
2. End-to-end recognition route based on deep learning. This method transforms interference signals into time-frequency images or raw waveforms, directly fed into deep models such as convolutional neural networks and recurrent neural networks for automatic feature learning and classification. Its advantage lies in eliminating the need for laborious manual feature engineering, theoretically capable of mining deeper discriminative information; however, its performance is highly dependent on large-scale high-quality annotated data, while the cost of obtaining interference data is high and annotation is difficult. The performance degradation under small sample sizes has become a bottleneck in engineering applications.
1.3 Contributions and Work of the Paper
Based on the above background, this paper conducts a systematic interference identification experimental study, with main contributions including:
(1) A mathematical simulation model covering three major interference systems and nine typical interference types has been constructed, generating samples under the same signal-to-noise ratio and transmitting signal parameters to ensure fairness in performance comparison among different interference styles;
(2) A set of seven physical interpretable features including energy, bandwidth, time domain continuity, pulse along energy ratio, time width, frequency domain moment skewness, and local moment skewness have been designed, and their discriminatory mechanisms in various interference classifications are analyzed one by one;
(3) Two sets of comparative experiments were designed—decision tree identification experiments for nine single interference types and deep convolutional neural network identification experiments under small sample conditions (including twelve types, including composite interference), cross-validation was conducted from three dimensions: recognition accuracy, feature importance, and sample sensitivity.
2 Interference Signal Modeling and Experimental Configuration
2.1 For the sake of generality, in this paper, a linear frequency modulated (LFM) pulse is taken as the radar emission baseline signal, whose complex envelope can be expressed as:
where _f_ is the frequency modulation slope, _t_ is the pulse width, and _B_ is the signal bandwidth. The LFM signal, which combines distance resolution and Doppler tolerance advantages, is widely used in modern radar for pulse compression waveforms. Building various interferences based on it can most closely simulate real combat scenarios.
2.2 To ensure that the experiment can be reproduced and the results have statistical significance, the simulation parameters are uniformly set as follows: pulse width, bandwidth, sampling rate (meeting Nyquist sampling condition), and the signal-to-noise ratio is fixed at 0 dB (additive Gaussian white noise is superimposed). Each type of interference generates 1000 samples for decision tree experiments and 300 samples for deep network experiments.
2.3 Nine types of interference signal models are divided into three major systems according to the fundamental differences in interference generation mechanism and signal structure, with the core mechanisms and mathematical expressions as follows:
(I) Forwarding Deception (Classes 1 to 4). This category interferes with the radar signal intercepted by DRFM as the core device, delaying, weighting, and superimposing the captured signals before forwarding, forming a pseudo target that is difficult to distinguish from real targets at the radar receiving end:
Full Pulse Forwarding (Class 1): Single-time delay amplification of LFM signals is performed before forwarding, creating a single pseudo target in the range dimension;
Full Pulse Dense Forwarding (Class 2): Multiple delays and superpositions are performed, forming 3 to 6 dense pseudo target groups in the range dimension;
Intermittent Sampling Forwarding (Class 3): Signal segments of varying lengths are periodically sampled and alternately forwarded, producing dense pseudo targets through pulse compression mismatch, balancing coherence and concealment;
Partial Pulse Dense Forwarding (Class 4): Density forwarding is performed within random time intervals within the pulse width, covering only part of the pulse, further enhancing anti-recognition capabilities.
(II) Noise Modulation Class (Class 5). Represented by intelligent noise, its core idea is to multiply the coherent signal transmitted by DRFM with random noise for modulation, making the interference both "coherent" (capable of passing through radar matched filtering) and "random", making it one of the most threatening interference systems currently available.
(III) Overpowering Class (Classes 6 to 9). These classes aim to destroy the radar detection capability, "burying" target echoes in the time or frequency domain:
Noise Frequency Modulation (Class 6): Bandwidth noise is generated by a frequency modulator to produce frequency modulation interference covering the entire bandwidth, with instantaneous frequency continuously changing;
Bandwidth Overpowering (Class 7): Gaussian white noise is directly injected into the receiving bandwidth, resulting in the most flat spectrum;
Sweep Frequency Interference (Class 8): Instantaneous frequency is scanned cyclically over time, achieving "energy focusing" type time-varying overpowering;
Comb Filter Spectrum Interference (Class 9): A combination of multiple narrow-band carriers is superimposed, forming a comb-like multi-peak structure in the frequency domain, selectively suppressing multiple critical frequencies.
3. Feature Extraction and Classification Methods
3.1 Seven-Dimension Multi-Domain Feature Design
The core contradiction in feature extraction lies in preserving sufficient discriminative information for categories while ensuring clear physical meanings and simple computations. This paper designs seven-dimensional features from three dimensions: time domain, frequency domain, and time domain statistics:
(1) Energy feature (F1): The power occupation pattern difference between jamming styles is significant—pressure jamming continuously radiates at full power with the highest energy; forward jamming only has energy during pulse intervals, which is overall lower. This feature is the first layer of "rough screening" indicator.
(2) Bandwidth feature (F2): Measuring the number of frequencies above the peak by about 30% (about -10 dB). Forward jamming bandwidth concentrates in the LFM main lobe range; noise modulated bandwidth significantly expands; comb-like spectrum consists of a few narrow-band carriers, with the widest bandwidth.
(3) Time domain continuity coefficient (F3): Measuring the proportion of sampling points below an adaptive threshold (determined by statistical quantities at the signal tail), reflecting the degree of signal sparsity. Intermittent sampling forward and partial pulse forward significantly reduce due to the "intermittent" nature of the signal.
(4) Sampling window mean (F4): Half of the sum of the average amplitudes at the head and tail of a pulse, characterizing the energy distribution around the leading and trailing edges of a pulse. Dense forward and backward have energy, while intermittent sampling forward only has signal in the middle segment, which can be used to distinguish between them.
(5) Time width feature (F5): Time span above a threshold of 0.1. Pressure jamming is not pulse-type signals, and their time width is generally smaller.
(6) Frequency domain moment skewness (F6): The normalized third central moment of the spectrum, reflecting the symmetry of the spectrum. Broadband suppressed spectrum has symmetry (skewness close to 0), sweep-frequency interference spectrum is tilted (negative skewness), and comb-like spectrum's multi-peak structure leads to positive skewness.
(7) Local moment skewness (F7): Calculating the mean skewness of each sub-segment after spectral division, capturing the local shape differences of the spectrum, used to refine the internal boundaries of suppressed classes.
The seven-dimensional features are progressively refined from coarse to fine, from the overall to the local level, forming a hierarchical structure for discrimination: energy and time width complete the first layer of broad category division, bandwidth and continuity coefficients complete the second layer of subdivision, and skewness class features are used for the most difficult classification within suppressed and forward categories.
3.2 Decision Tree-Based Fast Classification A decision tree is a tree-based classification model based on information gain, with its core advantages being: (1) It does not rely on assumptions about feature distribution and has strong adaptability to nonlinear discriminative boundaries; (2) The training and inference overhead is minimal, with microsecond-level inference speed meeting real-time adversarial needs; (3) The model structure can be "read" directly, and the split rules inherently possess interpretability. In this paper, the classifier regression tree (CART) is employed, with a 70%/30% split ratio for training and testing sets, and stability is evaluated through 10-fold cross-validation.
3.3 Convolutional Neural Network-Based Deep Learning Recognition To examine the performance of end-to-end learning paradigm in small sample and compound interference scenarios, this paper constructs a four-layer convolutional neural network (ECNN): The input is a three-channel time-frequency image generated by Short-Time Fourier Transform (STFT), corresponding to the real part, imaginary part, and modulus respectively. The network goes through layers of "convolution—batch normalization—ReLU—pooling," followed by dropout regularization, and finally outputs the category probability by fully connected layers. The network uses momentum-driven stochastic gradient descent optimization, with an initial learning rate of 0.001, maximum training rounds of 30, and dropout coefficient of 0.25.
4. Experimental Results and Analysis
4.1 Time-Domain and Frequency-Domain Waveform Analysis
Firstly, the time-domain waveforms of the nine types of interference are intuitively examined from the perspective of waveforms. Figure 1 presents the single sample time-domain waveforms of the nine types of interference:
Figure 1 Single sample time-domain waveforms of the nine types of interference
It can be observed that there is a clear differentiation in morphology: forwarding interference (class 14) retains the LFM pulse envelope characteristics, with dense forwarding (class 2) presenting multiple superimposed pulses; smart noise (class 5) maintains part of the LFM structure while adding random fluctuations; suppressed interference (class 69) exhibits continuous non-pulse-like waveforms, with broadband suppression (class 7) being the most flat, and comb-like spectrum (class 9) showing regular oscillations.
Figure 2 shows the corresponding frequency spectrum:
Figure 2 Frequency spectrum of the nine types of interference
The frequency spectrum of forwarding interference is concentrated within the main lobe bandwidth of LFM (±5 MHz); noise modulation has the widest and nearly flat spectrum; broadband suppression presents uniform white noise characteristics; sweep interference's frequency spectrum shows periodic fluctuations; comb-like spectrum displays a clear multi-peak structure, with the number of peaks equal to the number of carriers. The differences in waveform and spectrum at the physical level provide a basis for feature extraction.
4.2 Feature Distribution Analysis
Furthermore, statistical analysis is conducted on the distribution of seven-dimensional features across various interference categories to verify the effectiveness of the feature design.
Energy characteristics (Figure 3) exhibit a significant layered structure: suppressive interference, due to continuous full power radiation, has the highest energy; forwarding interference, only having energy during pulses and overall being lower, is centered by the multiple-carrier superposition. Energy characteristics can achieve large-scale classification with nearly zero computational cost.
Figure 3 Energy characteristic distribution
Bandwidth characteristics (Figure 4) are one of the most visually distinguishable features: forwarding bandwidth concentrates at 710 MHz (near the original LFM bandwidth), noise modulation can reach ultra-wideband levels of 1418 MHz, while comb spectrum bandwidth is the narrowest (less than 2 MHz), as it consists only of a few narrow-band carriers.
Figure 4 Bandwidth characteristic distribution
The time-domain continuity coefficient (Figure 5) plays a crucial role within the relay class: full pulse relaying and dense relaying, due to the signal's predominantly non-zero duration, have larger values; intermittent sampling and partial pulse relaying, however, significantly reduce these values due to their intermittency characteristics—this characteristic effectively widens the interclass distance within the relay class.
Figure 5 Continuity coefficient distribution
The Sample Window Mean (Figure 6) is specifically designed to distinguish between dense forwarding and intermittent sampling forwarding: the former has energy both before and after the pulse, with larger values; the latter only has signal in the middle segment, with smaller values. The two achieve nearly complete separability.
Figure 6: Mean distribution of sampling window
Time width features (Figure 7) are effective for suppressive types: the time width of suppressive types is generally small (<5 μs), as they are not pulse-type signals. The ambiguity noise, due to the inclusion of LFM components, has a slightly larger width, forming an additional discriminative dimension.
Figure 7: Time width distribution
The kurtosis of the frequency domain (Figure 8) is a key factor in suppressing internal subdivisions: broadband suppressed spectrum symmetry (Kurtosis close to 0), swept-frequency interference spectrum tilt (negative kurtosis), and comb-like spectrum multi-peak structure leading to positive kurtosis, which naturally separates along the kurtosis axis.
Figure 8: Moment Skewness Distribution
Local moment skewness (Figure 9) further refines the boundary: Broadband suppression shows consistent performance across sub-segments (low and stable values), while sweep frequency interference increases segment differences due to frequency shifts (higher values), effectively aiding in distinguishing between broadband suppression and sweep frequency interference.
Figure 9: Local moment skewness distribution
4.3 Decision Tree Identification Performance on Nine Types of Single Interference Tasks, the decision tree achieved an overall identification accuracy of 83.8%, and the results for each category are as shown in the table below:
Interference category
Accuracy identification
Main confusion object
1 Full pulse forwarding
92.3%  
A small number of misjudgments are mistaken for dense forwarding.
2. Density forwarding
82.3%  
Confused with full-pulse forwarding and intermitent sampling.
3 Intermittent Sampling Forwarding
64.3%  
Misinterpreted as partial pulse forwarding or agile noise.
4 Partial Pulse Forwarding
86.3%  
The main misjudgment was intermittent sampling.
5. Skilled Noise
45.3%  
The most challenging to classify, easily mixed in with forwarding or suppression categories.
6 Noise Frequency Modulation
93.3%  
Rare misjudgments
7 Broadband suppression
96.7%  
Nearly perfect classification
8 Frequency sweep interference
95.0%  
Rare misjudgments
9. Comb spectrum
98.7%  
The most recognizable
The confusion matrix of Figure 10 confirms the above-mentioned pattern:
Figure 10 Confusion Matrix
Figure 10 Decision Tree Classification Confusion Matrix
Pressure-based (Classes 6 to 9) show that diagonal elements are significantly larger than non-diagonal ones, indicating that artificial features have excellent discrimination capabilities, with recognition rates consistently between 93% and 99%. In the case of forwarding (Class 14), there is a certain degree of confusion among them, primarily due to their LFM derivative signals and highly similar time-frequency structures. The "smart noise" (Class 5) is the most challenging category due to its simultaneous characteristics of forwarding coherence and random noise, accounting for 45.3% of all categories.
4.4 Feature Importance Analysis Based on the Seven-Dimensional Feature Importance Ranking Calculated by the Contribution of Split Nodes, as shown in Figure 11:
Figure 11 Feature Importance
Figure 11: Seven-dimensional feature importance ranking
The results show that the order of importance is: energy > frequency domain moment skewness > sampling window mean > continuity coefficient > local moment skewness > time width > bandwidth. This result contains a "counterintuitive" phenomenon: the bandwidth characteristic has the strongest discriminative power at the level of human observation, but its importance is actually the lowest in decision trees. The reasonable explanation is that features such as energy and moment skewness have already included some discrimination information about bandwidth in the level of information gain. Decision trees choose the combination of features with "higher information content" based on the principle of maximizing information gain, thus losing the necessity for independent splitting in the tree structure. This phenomenon suggests that feature importance is not inherent to the features themselves, but depends on the redundancy structure within the feature set and the classification algorithm's splitting strategy. Human-designed features should not be solely based on the criterion of "single-dimensional separability."
In the twelve-class (including compound interference) task, the overall accuracy (OA) and average accuracy (AA) of ECNN change as shown in Figure 12:
Figure 12 ECNN recognition performance
The OA/AA curve of ECNN on the twelve-class task
Training sample proportion
Overall accuracy OA
Average accuracy AA
2%
47.1%
47.2%
4%
53.0%
53.1%
6%
57.9%
58.0%
8%
58.8%
58.9%
It can be seen that OA and AA increase steadily with increasing training sample proportion, from 47.1% at a 2% training sample proportion to 58.9% at an 8% training sample proportion; and OA and AA almost coincide, indicating balanced recognition across various categories without obvious long tail difficult cases. It is worth noting that ECNN improves rapidly in the low sample region of 2%~4%, but tends to saturate in the 6%~8% interval—this is directly related to the overall sample size and network capacity per class.
4.6 Comparison Analysis of Two Approaches
The results from sections 4.3 and 4.5 are juxtaposed to reveal a significant methodological conclusion: decision trees based on artificial features outperform end-to-end deep networks (up to 58.9%) in twelve categories tasks, significantly so (83.8%). The differences stem from three aspects: (1) the complexity of the task—the twelve categories include four types of compound interferences, with inter-class overlap significantly higher than in nine single interference categories; (2) the adaptability of features—artificial features have highly refined the physical basis for distinguishing interference signals, while deep networks have not fully converged under limited sample conditions; (3) the condition of samples—decision trees use relatively ample 70% training samples, whereas deep networks are deliberately constrained to extremely small sample sizes between 2% and 8%. This comparison precisely corroborates the core judgment mentioned earlier: in deployment scenarios with scarce labeled data, the combination of physical interpretable features and traditional learning algorithms remains a reliable baseline.
5. Discussion
5.1 The most prominent challenge in the experiment on the identification of elusive noise is the difficulty in identifying elusive noise (category 5, 45.3%). Its identification difficulties have profound physical roots: elusive noise is essentially a mixture of "coherent forwarding" and "random modulation," with energy levels between forwarding and suppressing categories, and it oscillates in the seven-dimensional feature space due to its LFM structure in time domain and noise broadening in frequency domain. This indicates that for elusive noise, higher-order discriminative information needs to be introduced, such as intra-pulse phase microstructure, cyclic stationarity characteristics, or high-order cumulants, as supplements to the existing feature system.
5.2 Another key issue revealed by the deep network experiments is the category confusion of compound interferences. There is extensive overlap between the four types of compound interferences and their corresponding single interferences in time-frequency plots, making it difficult for deep networks to completely separate them. This suggests that actual systems should combine "interference identification" with "interference decomposition": blind source separation or dictionary learning should estimate the interference components first, then separately discriminate each component rather than directly facing the "mixture" for end-to-end classification.
5.3 Limitations of the Study
The study has the following limitations: First, the signal-to-noise ratio is fixed at 0 dB, not covering the performance degradation under low signal-to-noise ratio conditions; second, complex coupling factors such as multipath, coexistence of target echoes and interferences, and radar parameter abrupt changes are not considered; third, the sample size for deep network experiments is small, and the extrapolation of conclusions to larger datasets needs to be verified.
6 Conclusion
This paper focuses on the core issue of radar interference identification, completing systematic experimental research from interference modeling, feature design to classification recognition. The main conclusions are as follows:
(1) Based on the three major interference systems, nine types of interference models have been constructed, showing significant and controllable differences in time domain, frequency domain, and statistical characteristics, providing a comprehensive verification platform for recognition algorithms;
(2) The seven-dimensional multi-domain feature system realizes a hierarchical discrimination structure from "rough to fine, from overall to local," supporting decision trees to achieve an overall accuracy of 83.8% on nine tasks, with suppression class interference recognition rates stabilizing at over 93%, verifying the core value of physical interpretable features in interference recognition;
(3) Feature importance analysis reveals the counterintuitive rule that "the strongest unidimensional separability does not necessarily mean the highest classification value," with energy features leading the decision tree splitting with slight information advantage, providing new design guidelines for feature engineering;
(4) Deep networks show stable convergence on twelve composite interference tasks with increasing sample sizes, but their performance is limited under extreme small sample sizes, complementing traditional feature methods - one suitable for data-rich scenarios, the other for data-scarce scenarios.
7 Future Prospects
Looking towards practical application needs, future research can delve into the following four directions:
1. Combinatorial Interference Decomposition and Identification: Construct a two-tier recognition architecture of "component decomposition + class-by-class discrimination" to address multi-interference scenarios;
2. Robustness to Low Signal-to-Noise Ratio: Systematically evaluate the performance degradation curves of various methods within the range of -10 dB~10 dB signal-to-noise ratio, exploring means to enhance features for weak interferences;
3.
Real-time and lightweight deployment: While maintaining the advantages of decision tree inference speed, real-time inference of deep networks on embedded platforms is achieved through quantization, pruning, and knowledge distillation.
Feature fusion strategy: The study explores the integration architecture of artificial features with deep network deep features in the later stage, balancing interpretability and recognition accuracy to form a hybrid intelligent recognition system of "expert knowledge + data-driven".
References: [1] Zhang Guoyi, et al. Jiebian radar anti-interference and signal processing technology [M]. Beijing: Defense Industry Press.
[2] Liu Zhong. Research on linear frequency modulation pulse compression radar interference technology based on DRFM [D]. Changsha: National University of Defense Technology, 2004.
[3] Wang Xuesong, Liu Jin, Danmei, et al. Method for generating radar decoy targets based on interruptive sampling forwarding interference [J]. System Engineering and Electronic Technology.
[4] Zhao Guoqing. Radar countermeasures principles [M]. Xi'an: Xi'an University of Electronic Science and Technology Press, 1999.
[5] Sparrow M J, Cikalo J. ECM techniques to counter pulse compression radar [R]. United States Patent.
[6] Li N, Zhang Y. A survey of radar ECM and ECCM [J]. IEEE Transactions on Aerospace and Electronic Systems.
[7] Chen Boxiao. Modern radar system analysis and design [M]. Xi'an: Xi'an University of Electronic Science and Technology Press, 2012.
This article is written based on the open interference identification simulation experiment framework, all experimental data and results figures can trace back to the corresponding simulation projects, welcome readers to discuss and exchange in the comments section.

## Images

![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](01.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](02.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](03.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](04.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](05.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](06.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](07.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](08.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](09.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](10.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](11.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](12.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](13.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](14.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](15.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](16.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](17.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](18.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](19.jpg)
![Intelligent Identification of Radar Interference Signals in Complex Electromagnetic Environments - A Systematic Experimental Study Based on Nine Category Interference Modeling and Multi-Domain Feature Analysis [with MATLAB Code]](20.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
