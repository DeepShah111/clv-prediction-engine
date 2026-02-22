# ==============================================================================
# evaluation.py
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error
from src.config import logger

def evaluate_and_plot(model, champion_name: str, X_test: pd.DataFrame, y_test: pd.Series):
    logger.info("[8/8] Evaluating Business Impact...")
    preds = model.predict(X_test)
    
    # Accuracy Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    wape = np.sum(np.abs(y_test - preds)) / np.sum(y_test)
    
    print(f"\n FINAL METRICS ({champion_name}) <<<\n   RMSE: ${rmse:.2f}\n   WAPE:  {wape:.2%}")
    
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=preds, alpha=0.6, color='purple')
    plt.plot([0, y_test.max()], [0, y_test.max()], 'r--')
    plt.title(f'Actual vs Predicted Spend (WAPE: {wape:.1%})')
    plt.savefig('artifacts/accuracy_check.png', bbox_inches='tight')
    plt.close()

    # Plot 2: Business Lift Chart
    eval_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds}).sort_values(by='Predicted', ascending=False)
    eval_df['Cum_Actual'] = eval_df['Actual'].cumsum() / eval_df['Actual'].sum()
    eval_df['Cum_Population'] = np.linspace(0, 1, len(eval_df))
    
    plt.figure(figsize=(10, 6))
    plt.plot(eval_df['Cum_Population'], eval_df['Cum_Actual'], label='Champion Model', color='green', lw=3)
    plt.plot([0, 1], [0, 1], 'r--', label='Random Baseline')
    plt.title('Business Lift Analysis (Gain Chart)')
    plt.legend()
    plt.savefig('artifacts/business_lift.png', bbox_inches='tight')
    plt.close()
    
    # Plot 3: Feature Importance (The "Why")
    try:
        estimator = model.named_steps['regressor'] if hasattr(model, 'named_steps') else model
        
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
            if hasattr(X_test, 'columns'):
                features = X_test.columns
            else:
                features = [f"Feature {i}" for i in range(len(importances))]
                
            fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10) # Top 10
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
            plt.title(f'Top 10 Feature Importance: {champion_name}')
            plt.xlabel('Relative Importance')
            plt.ylabel('')
            plt.tight_layout()
            plt.savefig('artifacts/feature_importance.png', bbox_inches='tight')
            plt.close()
            logger.info("Feature importance plot generated.")
        else:
            logger.warning("Champion model does not support feature importances. Skipping plot.")
            
    except Exception as e:
        logger.error(f"Could not generate feature importance plot: {e}")

    logger.info("Evaluation plots saved to artifacts/ folder.")

def save_model(model, feature_names: list):
    bundle = {
        'model': model,
        'feature_names': feature_names,
        'timestamp': pd.Timestamp.now(),
        'version': '1.2.0'
    }
    joblib.dump(bundle, 'artifacts/clv_champion_bundle.pkl')
    logger.info("Production Bundle Saved to artifacts/clv_champion_bundle.pkl")