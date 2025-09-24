

PREDICT_EXAMPLES = """
(EXAMPLE1)
Features: Granulocyte-to-lymphocyte ratio is 1.44, D-dimer is 0.19, chemotherapy, catheterization is CVC(Central Venous Catheter), no thoracic therapy, age at hospital is 29, platelet is 353.0, hemoglobin is 138.0, BMI is 18.83, gender is male, history of previous catheterization, no history of previous cather related thrombosis, no lung cancer, no gastric cancer, lymphoma, no gynecologic tumors, no urologic tumors.

Disease Prediction: no catheter-related thrombosis 

Explanation:Low Thrombotic Activity:
Normal D-dimer (0.19 mg/L): This value is well below the typical threshold for thrombosis (usually >0.5 mg/L), indicating no active clot formation or breakdown.

Normal Granulocyte-to-Lymphocyte Ratio (GLR = 1.44): Suggests no significant systemic inflammation, which can predispose to clotting.

Demographic and Physiologic Factors:
Young Age (29 years): Younger individuals generally have lower baseline thrombosis risk due to better endothelial health and fewer comorbidities.

Low BMI (18.83): Underweight/near-underweight BMI is associated with reduced thrombosis risk compared to obesity, which promotes hypercoagulability.

Absence of High-Risk Comorbidities:
No history of CRT or other thrombotic events, suggesting inherent resilience to catheter-induced clotting.

No lung, gastric, gynecologic, or urologic cancers, which are strongly linked to hypercoagulability. While lymphoma carries some thrombosis risk, its contribution here may be mitigated by effective chemotherapy or disease control.

Chemotherapy: While some agents increase thrombosis risk, the regimen used here may not be highly prothrombotic, or its effects are counterbalanced by the patient’s protective factors.

Laboratory Values:
Normal Platelets (353 x10⁹/L): Mild elevation is unlikely to drive clotting.
Normal Hemoglobin (138 g/L): No polycythemia or anemia, which can influence thrombosis risk.

Conclusion: Despite the presence of a CVC and lymphoma (both moderate risk factors), the patient’s youth, low BMI, absence of inflammation/thrombosis biomarkers, and lack of additional prothrombotic conditions (e.g., obesity, active cancer-related hypercoagulability) collectively protect against CRT. The normal D-dimer is particularly reassuring, reflecting no ongoing clotting activity.

(EXAMPLE2)
Features: Granulocyte-to-lymphocyte ratio is 2.73, D-dimer is 0.1, chemotherapy, catheterization is PICC(Peripherally Inserted Central Catheter), no thoracic therapy, age at hospital is 30, platelet is 267.0, hemoglobin is 108.0, BMI is 26.04, gender is female, no history of previous catheterization, no history of previous cather related thrombosis, no lung cancer, no gastric cancer, no lymphoma, no gynecologic tumors, no urologic tumors.

Disease Prediction: catheter-related thrombosis

Explanation:Catheter-Related Factors (PICC Line):
PICC lines directly increase thrombosis risk by causing endothelial injury, blood flow disruption, and activation of coagulation at the catheter site. This is the most significant contributor to CRT in this case.

Chemotherapy-Induced Hypercoagulability:
Chemotherapy agents are known to promote thrombosis by damaging the endothelium, increasing procoagulant factors, or reducing anticoagulant proteins. Even without a specific cancer diagnosis listed here, the prothrombotic effects of chemotherapy alone heighten CRT risk.

Inflammation (Elevated Granulocyte-to-Lymphocyte Ratio):
A GLR of 2.73 suggests systemic inflammation, which is linked to thrombosis via cytokine release (e.g., TNF-α, IL-6) and tissue factor activation. Chemotherapy itself can drive inflammation, exacerbating this risk.

Overweight Status (BMI 26.04):
Obesity is associated with chronic inflammation and venous stasis, both of which may compound thrombosis risk, particularly in the setting of a PICC line.

Anemia (Hemoglobin 108 g/L):
Chronic anemia, potentially due to chemotherapy or underlying illness, may alter blood viscosity and flow dynamics, contributing to a hypercoagulable state.

Normal D-Dimer Limitations:
The normal D-dimer (0.1 µg/mL) does not rule out CRT, as localized or small thrombi may not elevate systemic fibrin degradation products significantly.
Normal Platelets (267 ×10³/µL): Rules out thrombocytosis as a contributor.

Age/Gender: While younger age and female sex are generally lower-risk for VTE, chemotherapy and catheterization override these protective effects.

Conclusion:
The PICC line and chemotherapy are the primary drivers of CRT in this patient, with inflammation, overweight status, and anemia acting as contributing factors."""

