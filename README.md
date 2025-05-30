# BaekKhani_TripPurposeInferenceML
Codes for the paper: Navigating Machine Learning for Travel Behavior Analysis: A Comprehensive Guide to Inferring Trip Purposes Using Transit Survey and Automatic Fare Collection Data Fusion


Rounds Explanation:

Round 1: explored multiple categorical vars consolidations with OHE with 6-level target with obj=mean(acc,wf1)-0.05 zeropreds
Round 2: same with below Round B but obj=adj_accuracy and for RF, infinite depth option or (depth=200) has been removed, and now trials store overall accuracy and work_f1 regardless of the objective set

(Deprecated intermediate rounds between Rounds 1 and 2)
A: explored binary consolidations with all cat encoding with 3-level target (W/H/O) and extended range of denom and encdim; obj=mean(acc,wf1)
B: Finalized parameter spaces and employed 4-level target(W/H/O/Shop) but until this round; obj=wf1; has some intermediate results before final hp spaces fixed