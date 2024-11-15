import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score
from models import DyGED_NA, DyGED_CT, DyGED_NL, DyGED


# Φόρτωση των αρχείων που έχουν δημιουργηθεί από το custom_data_format.py
# Αυτά περιλαμβάνουν τα tensors για τα χαρακτηριστικά κόμβων (X), τις γειτνιάσεις (G) και τα labels (y_true)
G = torch.load('results/G_custom.pt')
X = torch.load('results/X_custom.pt')
y_true = torch.load('results/y_true_custom.pt')

# Αρχικοποίηση του πλήθους κόμβων (num_nodes) με βάση τις διαστάσεις του πίνακα γειτνίασης G
num_nodes = G.shape[1]

# -----------------------------------------------------------------------------------#
# Συνάρτηση για το διαχωρισμό των δεδομένων σε train και test sets χωρίς ανακάτεμα
def split_data(X, G, y_true, train_ratio):
    num_snapshots = X.shape[0]  # Πλήθος στιγμιότυπων (snapshots)
    train_size = int(num_snapshots * train_ratio)  # Υπολογισμός μεγέθους train set

    # Διατήρηση της χρονικής αλληλουχίας, αποφεύγοντας το ανακάτεμα
    X_train, X_test = X[:train_size], X[train_size:]
    G_train, G_test = G[:train_size], G[train_size:]
    y_train, y_test = y_true[:train_size], y_true[train_size:]

    return X_train, G_train, y_train, X_test, G_test, y_test

# -----------------------------------------------------------------------------------#

# Ρυθμίσεις εκπαίδευσης
train_ratio = 0.65  # Ποσοστό των δεδομένων για εκπαίδευση
learning_rate = 0.005  # Ρυθμός εκμάθησης για τον optimizer
dropout_rate = 0.02  # Dropout για την αποφυγή υπερεκμάθησης
num_epochs = 100  # Αριθμός επαναλήψεων εκπαίδευσης
num_hidden = 64  # Μέγεθος των κρυφών επιπέδων στα δίκτυα

# Χωρισμός δεδομένων σε train και test sets
X_train, G_train, y_train, X_test, G_test, y_test = split_data(X, G, y_true, train_ratio)

# Μετατροπή του y_test σε numpy array για χρήση στον υπολογισμό του AUC και του F1
y_test_np = y_test.cpu().detach().numpy()

# Εμφάνιση των διαστάσεων των train και test sets
print(X_train.shape)
print(X_test.shape)

# Λίστα με τα μοντέλα που θα εκπαιδευτούν και αξιολογηθούν
models = {
    "DyGED_CT": DyGED_CT(adj_size=num_nodes, n_feature=1, n_hidden_gcn=(num_hidden,), n_hidden_mlp=(num_hidden,),
                         n_output=1, k=0, dropout=dropout_rate, pooling_key='expert'),
    "DyGED_NL": DyGED_NL(adj_size=num_nodes, n_feature=1, n_hidden_gcn=(num_hidden,), attention_expert=1,
                         n_hidden_mlp=(num_hidden,), n_output=1, k=0, dropout=dropout_rate, pooling_key='expert'),
    "DyGED_NA": DyGED_NA(adj_size=num_nodes, n_feature=1, n_hidden_gcn=(num_hidden,), n_hidden_lstm=(num_hidden,),
                         n_hidden_mlp=(num_hidden,), n_output=1, dropout=dropout_rate, pooling_key='expert'),
    "DyGED": DyGED(adj_size=num_nodes, n_feature=1, n_hidden_gcn=(num_hidden,), n_hidden_mlp=(num_hidden,),
                   n_hidden_lstm=(num_hidden,),
                   attention_expert=1, n_output=1, k=0, dropout=dropout_rate, pooling_key='expert')
}

# Εκπαίδευση και αξιολόγηση για κάθε μοντέλο
for model_name, model in models.items():
    print(f"\nTraining and evaluating {model_name}...")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Ορισμός του optimizer για το μοντέλο

    # Εκπαίδευση του μοντέλου για num_epochs επαναλήψεις
    for epoch in range(num_epochs):
        model.train()  # Θέτουμε το μοντέλο σε κατάσταση εκπαίδευσης
        optimizer.zero_grad()  # Αρχικοποίηση των gradients σε 0

        outs, _, _, _ = model(X_train, G_train)  # Πρόβλεψη του μοντέλου στο train set

        # Χρήση binary cross-entropy loss με sigmoid (BCEWithLogitsLoss συνδυάζει sigmoid + binary cross entropy)
        loss = nn.BCEWithLogitsLoss()(outs, y_train.view(-1, 1))
        loss.backward()  # Υπολογισμός του gradient
        optimizer.step()  # Ενημέρωση των παραμέτρων του μοντέλου

    # Αξιολόγηση στο test set
    model.eval()
    with torch.no_grad():
        outs_test, _, _, _ = model(X_test, G_test)  # Πρόβλεψη στο test set

        # Εφαρμογή της sigmoid στην έξοδο για τη μετατροπή σε πιθανότητες
        probabilities = torch.sigmoid(outs_test)
        probabilities_np = probabilities.cpu().detach().numpy()  # Μετατροπή σε numpy array

        # Υπολογισμός του AUC (Area Under Curve)
        auc_score = roc_auc_score(y_test_np, probabilities_np)
        print(f'AUC Score for {model_name}: {auc_score}')

        # Μετατροπή των πιθανοτήτων σε δυαδικές τιμές με κατώφλι 0.5 για τον υπολογισμό του F1 Score
        predictions_np = (probabilities_np >= 0.5).astype(int)
        f1 = f1_score(y_test_np, predictions_np)
        print(f'F1 Score for {model_name}: {f1}')
