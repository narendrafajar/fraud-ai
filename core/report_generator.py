import pandas as pd

def generate_report(df, output_path="analysis_result.xlsx"):

    total_transactions = len(df)
    total_anomaly = df["predicted_anomaly"].sum()
    total_risk_value = df[df["predicted_anomaly"] == 1]["amount"].sum()

    summary = pd.DataFrame({
        "Metric": [
            "Total Transactions",
            "Detected Anomalies",
            "Total Risk Amount"
        ],
        "Value": [
            total_transactions,
            total_anomaly,
            total_risk_value
        ]
    })

    with pd.ExcelWriter(output_path) as writer:

        df.to_excel(writer,
                    sheet_name="All Transactions",
                    index=False)

        df[df["predicted_anomaly"] == 1] \
            .to_excel(writer,
                      sheet_name="Detected Anomalies",
                      index=False)

        summary.to_excel(writer,
                         sheet_name="Executive Summary",
                         index=False)
