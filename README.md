<h1>XAI for IoT Systems</h1>

<h2>Datasets</h2>

<h3>(1)  MEMS datasets:</h3>
To build these datasets, an experiment was conducted in the motor testbed to collect machine condition data (i.e., acceleration) for different health conditions. During the experiment, the acceleration signals were collected from both piezoelectric and MEMS sensors at the same time with the sampling rate of 3.2 kHz and 10 Hz, respectively, for X, Y, and Z axes. Different levels of machine health condition can be induced by mounting a mass on the balancing disk, thus different levels of mechanical imbalance are used to trigger failures. Failure condition can be classified as one of three possible states - normal, near-failure, and failure.
Multiple levels of the mechanical imbalance can be generated in the motor testbed (i.e., more masses indicate worse health condition). In this experiment, three levels of mechanical imbalance (i.e., normal, near-failure, failure) were considered
Acceleration data were collected at the ten rotational speeds (100, 200, 300, 320, 340, 360, 380, 400, 500, and 600 RPM) for each condition, while the motor is running, 50 samples were collected at 10 s interval, for each of the ten rotational speeds. We use this same data for defect-type classification and learning transfer tasks.

<h3>(2) N-BaIoT dataset:</h3>
It was created to detect IoT botnet attacks and is a useful resource for researching cybersecurity issues in the context of the Internet of Things (IoT).
This data was gathered from nine commercial IoT devices that were actually infected by two well-known botnets, Mirai and Gafgyt.

Every data instance in the dataset has access to a variety of features. These attributes are divided into multiple groups:

A. Stream Aggregation: These functions offer data that summarizes the traffic of the past few days. This group's categories comprise:
H: Statistics providing an overview of the packet's host's (IP) recent traffic.
HH: Statistics providing a summary of recent traffic from the host (IP) of the packet to the host of the packet's destination.
HpHp: Statistics providing a summary of recent IP traffic from the packet's source host and port to its destination host and port.
HH-jit: Statistics that summarize the jitter of the traffic traveling from the IP host of the packet to the host of its destination.

B. Time-frame (Lambda): This characteristic indicates how much of the stream's recent history is represented in the statistics. They bear the designations L1, L3, L5, and so forth.

C. Data Taken Out of the Packet Stream Statistics: Among these characteristics are:

Weight: The total number of objects noticed in recent history, or the weight of the stream.

Mean: The statistical mean is called the mean.

Std: The standard deviation in statistics.

Radius: The square root of the variations of the two streams.

Magnitude: The square root of the means of the two streams.

Cov: A covariance between two streams that is roughly estimated.

Pcc: A covariance between two streams that is approximated.

The dataset consists of the following 11 classes: benign traffic is defined as network activity that is benign and does not have malicious intent, and 10 of these classes represent different attack tactics employed by the Gafgyt and Mirai botnets to infect IoT devices. 

1. benign: There are no indications of botnet activity in this class, which reflects typical, benign network traffic. It acts as the starting point for safe network operations.

2. gafgyt.combo: This class is equivalent to the "combo" assault of the Gafgyt botnet, which combines different attack techniques, like brute-force login attempts and vulnerability-exploiting, to compromise IoT devices.

3. gafgyt.junk: The "junk" attack from Gafgyt entails flooding a target device or network with too many garbage data packets, which can impair operations and even result in a denial of service.

4. gafgyt.scan: Gafgyt uses the "scan" attack to search for IoT devices that are susceptible to penetration. The botnet then enumerates and probes these devices in an effort to locate and compromise them.

5. gafgyt.tcp: This class embodies the TCP-based attack of the Gafgyt botnet, which targets devices using TCP-based exploits and attacks.

6. gafgyt.udp: The User Datagram Protocol (UDP) is used in Gafgyt's "udp" assault to initiate attacks, such as bombarding targets with UDP packets to stop them from operating.

7. mirai.ack: To take advantage of holes in Internet of Things devices and enlist them in the Mirai botnet, Mirai's "ack" attack uses the Acknowledgment (ACK) packet.

8. mirai.scan: By methodically scanning IP addresses and looking for vulnerabilities, Mirai's "scan" assault seeks to identify susceptible Internet of Things (IoT) devices.

9. mirai.syn: The Mirai "syn" attack leverages vulnerabilities in Internet of Things devices to add them to the Mirai botnet by using the SYN packet, which is a component of the TCP handshake procedure.

10. mirai.udp: Based on the UDP protocol, Mirai's "udp" attack includes bombarding targeted devices with UDP packets in an attempt to interfere with their ability to function.

11. mirai.udpplain: This class represents plain UDP assaults that aim to overload IoT devices with UDP traffic, causing service disruption. It is similar to the prior "udp" attack by Mirai.


<h2>How to run the program</h2>

Inside the MEMS or IoT folders, you will find:

1. Programs for metrics of each model used in this paper. Each one of these programs outputs the Accuracy, Precision, Recall and F1-score for the AI model.

2. A folder named feature importances, containing programs for ranking the importance of the features for different techniques of each model used in this paper. Each one of these programs outputs a bar graph with the x-axis representing feature names and y-axis represeting the importance scores.

<h2>Sample Evaluation Results</h2>

<h4>SHAP Global Summary Plot</h4>

![W2 Mems](https://github.com/agummadi1/XAI_for_IoT_Systems/assets/154301345/536a8f0d-694e-4c52-8300-708821adba74)

This is a representation of the SHAP (SHapley Additive exPlanations) values, which measure each feature's contribution to a machine learning model's prediction. Every horizontal bar is an unique feature, and its length signifies the mean absolute SHAP value, which reflects the average effect of the feature on the output magnitude of the model. Each bar's color—magenta for "Failure," green for "Normal," and blue for "Near-failure"—represents the percentage of each feature's influence on the various prediction classifications. For example, feature 'z' significantly affects the "Normal" class, but features 'x' and 'y' appear to have varying effects on both the "Normal" and "Failure" predictions. 

Feature Z appears to be most influential in predicting all the 3 labels, suggesting it helps in recognizing normal operational states, detecting situations that are close to failure but not yet critical and also identifying potential failures.

<h4>Feature Importance using Local Interpretable Model-Agnostic Explanations (LIME)</h4>

![Feature_imp_lime](https://github.com/agummadi1/XAI_for_IoT_Systems/assets/154301345/a4568cb1-4f3b-46bc-9e86-7093d2e65dc2)

The prediction probabilities for the three classes (designated as 0, 1, and 2) are displayed with the feature contributions that went into the forecast for each class, with each row representing a distinct case.
Understanding the machine learning model's reasoning behind certain predictions is made easier with the help of this image. This is particularly helpful for high-stakes applications like finance or healthcare, where decision-making must be transparent and comprehensible.

<h4>Performance Metrics with 115 Features of IoT device 9</h4>

![device9_metrics](https://github.com/agummadi1/XAI_for_IoT_Systems/assets/154301345/00bed78a-6732-46d0-80b7-f24aeb75c0b2)

Using a variety of machine learning models, the table gives a thorough summary of the anomaly detection outcomes for 1 of the 9 IoT devices. Performance measures such as accuracy, precision, recall, and F-1 score are displayed in the columns, with each row representing a distinct model. Notably, all of the models—Decision Tree, Random Forest, Bagging, Blending, Stacking, and Voting—show consistently high scores on all measures, with recall, F-1 Score, and accuracy and precision all approaching 0.99. These algorithms demonstrate remarkable efficacy in precisely detecting data anomalies. On the other hand, the ADA model has somewhat lower scores, suggesting a less robust anomaly detection system. While the performance of the MLP and DNN models varies, the SVM model performs moderately. 

<h4>Feature Importance for device 8 using Permutation Feature Importance (PFI) with different models></h4>

![d7_MLP_pfi_feature_importance](https://github.com/agummadi1/XAI_for_IoT_Systems/assets/154301345/6415d163-3022-459b-9b6d-789a0f63e241)
![d7_random_forest_pfi_feature_importance](https://github.com/agummadi1/XAI_for_IoT_Systems/assets/154301345/e31720d9-ea80-4f13-9994-9b892b82c98a)
![d7_decision_tree_pfi_feature_importance](https://github.com/agummadi1/XAI_for_IoT_Systems/assets/154301345/df21c465-16ff-433f-a2c9-50587aa4ce45)
![d7_ada_pfi_feature_importance](https://github.com/agummadi1/XAI_for_IoT_Systems/assets/154301345/63c2f758-0aa3-4b1a-9bd1-536123f9fc82)



