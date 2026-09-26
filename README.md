# SmartStock Magallanes: High-Volatility Predictive Logistics & Freshness Classification

Final project for the Building AI course

## Summary
SmartStock Magallanes is an advanced, deployment-ready machine learning framework tailored for neighborhood minimarkets experiencing extreme sub-Antarctic weather variations and high financial volatility. By deploying an explicit Linear Regression equation for daily bakery forecasting and a K-Nearest Neighbor (KNN) vector model for biological spoilage detection, this system completely replaces intuitive "al ojo" operations with data-driven automated retail intelligence.

## Background
Operating a retail grocery store in the Magallanes region (Punta Arenas) introduces critical operational anomalies that traditional, linear retail models cannot handle.
* **The High-Volatility Revenue Gap:** Daily revenue fluctuates aggressively by up to 40% (dropping from \$1,000,000 CLP to \$600,000 CLP with luck), rendering fixed-volume stock ordering financially hazardous.
* **The Perishable Waste Paradox:** Standard bakery goods (bread) suffer a baseline daily volume of 8 kg on stable days, but unexpected atmospheric drops cause rapid 60% demand collapses, turning unsold stock into direct net losses.
* **The Construction Site Micro-Traffic:** Nearby residential infrastructure projects inject highly volatile lunch-hour demand. While crews buy empanadas rather than raw bread loaves, their sudden absence causes massive, unpredictable shifts in daily baked goods turnover.
* **Severe Biological Food Risks:** The complete absence of structured stock tracking has led to critical food quality failures, allowing high-moisture items like yogurt cakes and carrot breads to remain unmonitored for over 7 consecutive days, resulting in severe fungal growth (mold).

## How is it used?

The application functions as a dual-engine operations assistant running on a low-latency digital dashboard:
```
[Meteorological APIs] ----+|--> [Linear Regression Engine] --> Daily Bread & Empanada Target (kg)[Construction Logs] ------+
[Batch Scan Entry] ----------> [KNN Freshness Classifier] --> Visual Freshness Alert (Risk 0-2)
```
1. **Predictive Ordering Layer (17:00 PM):** Ingests real-time local wind, snow, and rain alerts alongside construction data to define exact baking limits for the next morning.
2. **Biological Tracking Layer (Morning Intake):** Replaces physical oversight by logging fresh bakery batches into an automated vector matrix that actively pushes degradation warnings to staff.

## Data sources and AI methods

### 1. Daily Demand Prediction (Linear Regression Model)
To handle the 60% demand collapse during Magallanes storms, we establish an explicit linear function to output the optimal bread/bakery volume (\(Y_{vol}\)):

\[\hat{y}_{vol} = a + c_1x_1 + c_2x_2\]

Where:
* **a (Intercept) = 8.0:** The baseline volume in kilograms during a standard, undisturbed operating day.
* **c₁ (Weather Coefficient) = -4.8:** A negative coefficient triggering a 60% reduction (dropping demand by 4.8 kg) when extreme wind, rain, or snow protocols are active (x₁ = 1).
* **c₂ (Crew Coefficient) = +2.5:** An empirical adjustment coefficient reflecting the presence of active construction crews (x₂ = 1).

### 2. Spoilage Detection & Freshness Tracking (KNN Classification)
To eliminate the 7-day severe molding crisis in the pastry showcases, we deploy a **K-Nearest Neighbor (K=3)** classifier. Pastry items are evaluated as abstract multi-dimensional feature vectors:

\[X_{pastry} = [\text{Days Elapsed}, \text{Relative Humidity}, \text{Product Category Type}]\]

The model computes the exact Euclidean distance against historical lab parameters to output a deterministic risk classification label:
* **Class 0 (Green):** Safe for consumption (<3 days elapsed).
* **Class 1 (Yellow):** Critical Inspection Triggered (Days 4–5).
* **Class 2 (Red):** Mandatory Biological Disposal Active (    * Days 6–7 elapsed; absolute fungal risk).

```python
# Pure Python execution vector for the KNN distance tracking
import math

def calculate_spoilage_risk(new_batch, historical_sample):
    # Features: [Days, Moisture_Scale]
    distance = math.sqrt((new_batch - historical_sample)**2 + (new_batch - historical_sample)**2)
    return distance
```

## Challenges
The framework cannot mitigate direct external macroeconomic shocks, such as regional distributor delays cutting off the supply chain from central Chile. Crucially, because the store currently has zero historical data infrastructure, the system requires a strict **30-day Warm-Up Pipeline**. During this initial month, data must be manually recorded to stabilize the feature weights before the predictive outputs achieve high statistical precision.

## What next?
The framework can scale efficiently by deploying low-cost IoT digital weight scales under the bread and pastry display cases. This hardware expansion will feed real-time inventory depletion rates directly into the machine learning pipeline, enabling automated stock depletion tracking and generating instant purchase orders transmitted via automated communication APIs.

## Acknowledgments
* Algorithmic structures based on the machine learning principles of the *Elements of AI* curriculum by the University of Helsinki and Reaktor.
* Grounded in empirical operational metrics, micro-traffic flows, and localized biological storage realities from sub-Antarctic retail environments in Chilean Patagonia.
