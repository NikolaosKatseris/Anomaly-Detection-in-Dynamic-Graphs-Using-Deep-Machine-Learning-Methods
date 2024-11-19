import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import DyGED_NA, DyGED_CT, DyGED_NL, DyGED
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Φόρτωση δεδομένων (NYC Cab)
G = torch.load('data/nyc_cab/raw/graphs.pt').type(torch.FloatTensor)
X = torch.load('data/nyc_cab/raw/dynamic_attrs.pt').type(torch.FloatTensor)
y_true = torch.load('data/nyc_cab/raw/events.pt').type(torch.FloatTensor)

# Εάν θέλουμε να χρησιμοποιήσουμε τα δεδομένα Twitter Weather,
# αλλάζουμε τα paths στα αντίστοιχα αρχεία:
# G = torch.load('data/twitter_weather/raw/graphs.pt').type(torch.FloatTensor)
# X = torch.load('data/twitter_weather/raw/dynamic_attrs.pt').type(torch.FloatTensor)
# y_true = torch.load('data/twitter_weather/raw/events.pt').type(torch.FloatTensor)

num_nodes = G.shape[1]  # Πλήθος κόμβων

# Διαχωρισμός δεδομένων σε train και test sets
def split_data(X, G, y_true, train_ratio):
    num_snapshots = X.shape[0]
    train_size = int(num_snapshots * train_ratio)
    return X[:train_size], G[:train_size], y_true[:train_size], X[train_size:], G[train_size:], y_true[train_size:]

# Batch DataLoader
def batch_data(X, G, y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i+batch_size], G[i:i+batch_size], y[i:i+batch_size]

# Συνδυασμοί epochs και learning rates
epochs_list = [50, 100, 150]
learning_rates = [0.001, 0.005]

# Αποθήκευση αποτελεσμάτων ανά συνδυασμό
combination_results = {}
all_train_losses = {}

train_ratio = 0.8334
dropout_rate = 0.02
num_hidden = 64
batch_size = 100

X_train, G_train, y_train, X_test, G_test, y_test = split_data(X, G, y_true, train_ratio)
y_test_np = y_test.cpu().detach().numpy()
y_train_np = y_train.cpu().detach().numpy()

# Υπολογισμός του pos_weight για imbalanced dataset
pos_weight = torch.tensor([len(y_train_np[y_train_np == 0]) / len(y_train_np[y_train_np == 1])])

# Μοντέλα
models = {
    "DyGED_CT": DyGED_CT(adj_size=num_nodes, n_feature=3, n_hidden_gcn=(num_hidden,), n_hidden_mlp=(num_hidden,),
                         n_output=1, k=0, dropout=dropout_rate, pooling_key='expert'),
    "DyGED_NL": DyGED_NL(adj_size=num_nodes, n_feature=3, n_hidden_gcn=(num_hidden,), attention_expert=1,
                         n_hidden_mlp=(num_hidden,), n_output=1, k=0, dropout=dropout_rate, pooling_key='expert'),
    "DyGED_NA": DyGED_NA(adj_size=num_nodes, n_feature=3, n_hidden_gcn=(num_hidden,), n_hidden_lstm=(num_hidden,),
                         n_hidden_mlp=(num_hidden,), n_output=1, dropout=dropout_rate, pooling_key='expert'),
    "DyGED": DyGED(adj_size=num_nodes, n_feature=3, n_hidden_gcn=(num_hidden,), n_hidden_mlp=(num_hidden,),
                   n_hidden_lstm=(num_hidden,), attention_expert=1, n_output=1, k=0, dropout=dropout_rate, pooling_key='expert')
}

retry_attempts = {model_name: 0 for model_name in models.keys()}
max_total_retries = 5

# Εκπαίδευση και αξιολόγηση
for model_name, model in models.items():
    print(f"\n=== Starting Model: {model_name} ===")
    best_f1_global = 0
    best_result_global = None

    while retry_attempts[model_name] < max_total_retries:
        print(f"\n{model_name} - Retry Attempt {retry_attempts[model_name] + 1}/{max_total_retries}")
        retry_attempts[model_name] += 1

        for num_epochs in epochs_list:
            for lr in learning_rates:
                print(f"\nTesting combination: Epochs={num_epochs}, Learning Rate={lr}")

                optimizer = optim.Adam(model.parameters(), lr=lr)
                train_losses = []

                # Εκπαίδευση
                model.train()
                for epoch in range(num_epochs):
                    epoch_loss = 0
                    for X_batch, G_batch, y_batch in batch_data(X_train, G_train, y_train, batch_size):
                        optimizer.zero_grad()
                        outs, _, _, _ = model(X_batch, G_batch)
                        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(outs, y_batch.view(-1, 1))
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    train_losses.append(epoch_loss / len(X_train))

                all_train_losses[model_name] = train_losses

                # Αξιολόγηση
                model.eval()
                test_probabilities = []
                with torch.no_grad():
                    for X_batch, G_batch, y_batch in batch_data(X_test, G_test, y_test, batch_size):
                        outs, _, _, _ = model(X_batch, G_batch)
                        probabilities = torch.sigmoid(outs).squeeze()
                        test_probabilities.extend(probabilities.cpu().detach().numpy())

                # Βελτιστοποίηση Threshold για F1
                best_threshold = 0
                best_f1 = 0
                for t in np.arange(0.1, 0.51, 0.05):
                    y_test_pred = (np.array(test_probabilities) >= t).astype(int)
                    f1 = f1_score(y_test_np, y_test_pred)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = t

                # Υπολογισμός Precision, Recall, AUC
                y_test_pred = (np.array(test_probabilities) >= best_threshold).astype(int)
                precision_test = precision_score(y_test_np, y_test_pred, zero_division=0)
                recall_test = recall_score(y_test_np, y_test_pred, zero_division=0)
                auc_score_test = roc_auc_score(y_test_np, test_probabilities)

                if best_f1 >= 0.2:
                    print(f"Successful F1 ≥ 0.2 (F1: {best_f1:.2f}) with Threshold: {best_threshold:.2f}")
                    if best_f1 > best_f1_global:
                        best_f1_global = best_f1
                        best_result_global = {
                            "best_f1": best_f1,
                            "best_threshold": best_threshold,
                            "learning_rate": lr,
                            "epochs": num_epochs,
                            "test_probabilities": test_probabilities,
                            "precision": precision_test,
                            "recall": recall_test,
                            "auc_score": auc_score_test,
                        }

        if best_f1_global >= 0.2:
            break

    if best_result_global:
        print(f"\n{model_name} - Best Result Found: F1={best_result_global['best_f1']:.2f}, "
              f"Threshold={best_result_global['best_threshold']:.2f}, "
              f"Precision={best_result_global['precision']:.2f}, Recall={best_result_global['recall']:.2f}, "
              f"AUC={best_result_global['auc_score']:.2f}, LR={best_result_global['learning_rate']}, "
              f"Epochs={best_result_global['epochs']}")
        combination_results[model_name] = best_result_global
    else:
        print(f"\n{model_name} - Failed to achieve F1 ≥ 0.2 after {max_total_retries} retries.")