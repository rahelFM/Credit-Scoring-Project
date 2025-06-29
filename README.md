# Credit Scoring Business Understanding

## 1. Basel II Accord's Influence on Model Requirements

The Basel II Capital Accord's emphasis on rigorous risk measurement directly impacts our modeling approach in three key ways:

1. **Regulatory Compliance**: The Accord's Pillar 1 requires banks to maintain capital reserves proportional to their risk exposure. This necessitates models whose calculations can be clearly traced and validated by regulators.

2. **Supervisory Review (Pillar 2)**: Banking authorities must approve internal models, requiring complete documentation of:
   - Variable selection rationale
   - Model development methodology
   - Validation procedures
   - Performance tracking mechanisms

3. **Market Discipline (Pillar 3)**: Public disclosure requirements mean models must produce outputs that are both statistically sound and explainable to non-technical stakeholders.

These requirements make interpretability and documentation non-negotiable, as regulators must be able to audit and understand every modeling decision.

## 2. Proxy Variable Necessity and Risks

**Why a proxy is essential:**
- The eCommerce dataset lacks traditional credit performance data
- Behavioral patterns (RFM metrics) serve as the best available indicators
- Without a proxy, we cannot train a supervised learning model
- Basel II permits alternative data approaches when properly validated

**Potential business risks:**
1. **Misalignment Risk**: Purchase behavior may not perfectly correlate with repayment capacity
2. **Concept Drift**: Ecommerce patterns may change post-credit offering
3. **Segmentation Errors**: High-value shoppers ≠ good credit risks
4. **Regulatory Scrutiny**: Proxy justification must withstand audit

**Mitigation Strategies**:
- Conservative initial credit limits
- Rapid feedback loops when real repayment data emerges
- Regular model recalibration
- Human oversight for edge cases

## 3. Model Complexity Trade-offs

| Consideration          | Simple Model (Logistic Regression + WoE) | Complex Model (Gradient Boosting) |
|------------------------|------------------------------------------|-----------------------------------|
| **Interpretability**   | High - Clear feature weights             | Medium - Requires SHAP/LIME       |
| **Regulatory Approval**| Easier - Transparent calculations       | Harder - "Black box" concerns     |
| **Predictive Power**   | Lower - Linear assumptions              | Higher - Captures interactions    |
| **Implementation**     | Faster - Fewer hyperparameters          | Slower - Tuning intensive         |
| **Maintenance**        | Easier - Stable behavior                | Harder - Potential instability    |

**Recommended Approach**: Begin with interpretable models for initial regulatory approval while simultaneously developing complex models for comparison. Use model blending techniques if justified by significant performance improvements.

**Key Decision Factors**:
1. Regulatory environment strictness
2. Availability of model validation resources
3. Business risk appetite
4. Technical debt considerations
