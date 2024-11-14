import torch
import torch.nn as nn
import time
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from models import DyGED_NA, DyGED_CT, DyGED_NL, DyGED


# Φόρτωση αρχείων δεδομένων από τον καθορισμένο φάκελο
G = torch.load('data/nyc_cab/raw/graphs.pt')
G = G.type(torch.FloatTensor)

X = torch.load('data/nyc_cab/raw/dynamic_attrs.pt')
X = X.type(torch.FloatTensor)

y_true = torch.load('data/nyc_cab/raw/events.pt')
y_true = y_true.type(torch.FloatTensor)

#Αρχικοποίηση πλήθος κόμβων
num_nodes = G.shape[1]

# -----------------------------------------------------------------------------------#
# Διαχωρισμός δεδομένων σε train και test sets χωρίς ανακάτεμα
def split_data(X, G, y_true, train_ratio):
    num_snapshots = X.shape[0]
    train_size = int(num_snapshots * train_ratio)

    # Χωρισμός των δεδομένων χωρίς ανακάτεμα (διατήρηση χρονικής αλληλουχίας)
    X_train, X_test = X[:train_size], X[train_size:]
    G_train, G_test = G[:train_size], G[train_size:]
    y_train, y_test = y_true[:train_size], y_true[train_size:]

    return X_train, G_train, y_train, X_test, G_test, y_test

# -----------------------------------------------------------------------------------#

# Ρυθμίσεις εκπαίδευσης
train_ratio = 0.8334
learning_rate = 0.005
dropout_rate = 0.02
num_epochs = 100
num_hidden = 64

# Χωρισμός δεδομένων σε train και test sets
X_train, G_train, y_train, X_test, G_test, y_test = split_data(X, G, y_true, train_ratio)
y_test_np = y_test.cpu().detach().numpy()
print(X_train.shape)
print(X_test.shape)

# Λίστα με τα μοντέλα που θα εκπαιδευτούν και αξιολογηθούν
models = {
    "DyGED_CT": DyGED_CT(adj_size=num_nodes, n_feature=3, n_hidden_gcn=(num_hidden,), n_hidden_mlp=(num_hidden,),
                         n_output=1, k=0, dropout=dropout_rate, pooling_key='expert'),
    "DyGED_NL": DyGED_NL(adj_size=num_nodes, n_feature=3, n_hidden_gcn=(num_hidden,), attention_expert=1,
                         n_hidden_mlp=(num_hidden,), n_output=1, k=0, dropout=dropout_rate, pooling_key='expert'),
    "DyGED_NA": DyGED_NA(adj_size=num_nodes, n_feature=3, n_hidden_gcn=(num_hidden,), n_hidden_lstm=(num_hidden,),
                         n_hidden_mlp=(num_hidden,), n_output=1, dropout=dropout_rate, pooling_key='expert'),
    "DyGED": DyGED(adj_size=num_nodes, n_feature=3, n_hidden_gcn=(num_hidden,), n_hidden_mlp=(num_hidden,),
                   n_hidden_lstm=(num_hidden,),
                   attention_expert=1, n_output=1, k=0, dropout=dropout_rate, pooling_key='expert')
}

# Εκπαίδευση και αξιολόγηση για κάθε μοντέλο
for model_name, model in models.items():
    print(f"\nTraining and evaluating {model_name}...")

    # Χρονική μέτρηση από την αρχή της εκπαίδευσης μέχρι το AUC score
    start_time = time.time()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Εκπαίδευση του μοντέλου
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outs, _, _, _ = model(X_train, G_train)

        # Χρήση binary cross-entropy loss με sigmoid
        loss = nn.BCEWithLogitsLoss()(outs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()

    # Αξιολόγηση στο test set
    model.eval()
    with torch.no_grad():
        outs_test, _, _, _ = model(X_test, G_test)

        # Εφαρμογή της sigmoid στην έξοδο
        probabilities = torch.sigmoid(outs_test)
        probabilities_np = probabilities.cpu().detach().numpy()
        auc_score = roc_auc_score(y_test_np, probabilities_np)
        print(f'AUC Score for {model_name}: {auc_score}')
    # Υπολογισμός και εμφάνιση του συνολικού χρόνου
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time for training and evaluation of {model_name}: {total_time:.2f} seconds')