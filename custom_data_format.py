import torch
import os
import re

# Αρχικοποίηση των κόμβων
num_nodes = 500
# -----------------------------------------------------------------------------------#
# Διαδρομή προς τον φάκελο που περιέχει τα αρχεία
#folder_path = 'results/30_60_15_0.6_0.8_0.2_1'
#folder_path = 'results/150_100_15_0.6_0.8_0.2_1'
folder_path = 'results/500_60_15_0.6_0.8_0.2_1'
# Ανάγνωση όλων των διαθέσιμων αρχείων από το φάκελο
graph_files = [f for f in os.listdir(folder_path) if f.startswith('graph-') and f.endswith('.txt')]

# Ταξινόμηση των αρχείων βάσει αριθμού στιγμιότυπου
graph_files = sorted(graph_files, key=lambda x: int(re.search(r'graph-(\d+)', x).group(1)))

num_snapshots = len(graph_files)

# Δημιουργία του τανυστή G
G = torch.zeros((num_snapshots, num_nodes, num_nodes))

# Επεξεργασία κάθε διαθέσιμου αρχείου
for snapshot_idx, file_name in enumerate(graph_files):
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'r') as f:
        for line in f:
            u, v = map(int, line.strip().split())
            G[snapshot_idx, u, v] = 1
            G[snapshot_idx, v, u] = 1  # Μη κατευθυνόμενο γράφημα

# Αποθήκευση του τανυστή G
torch.save(G, 'results/G_custom.pt')
print("Tensor G has been saved as 'results/G_custom.pt'")
G = G.type(torch.FloatTensor)
# -----------------------------------------------------------------------------------#
# Δημιουργία του τανυστή X
X = torch.zeros((num_snapshots, num_nodes, 1))

# Ανάγνωση όλων των διαθέσιμων αρχείων κοινότητας
community_files = [f for f in os.listdir(folder_path) if f.startswith('communities') and f.endswith('.txt')]
community_files = sorted(community_files, key=lambda x: int(re.search(r'communities-(\d+)', x).group(1)))

# Επεξεργασία κάθε διαθέσιμου αρχείου
for snapshot_idx, file_name in enumerate(community_files):
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'r') as f:
        for line in f:
            community_id, nodes_str = line.strip().split('\t')
            community_id = int(community_id)
            nodes = eval(nodes_str)  # Μετατροπή της συμβολοσειράς λίστας σε λίστα

            # Για κάθε κόμβο στην κοινότητα, αποδίδουμε το community_id στο X
            for node in nodes:
                X[snapshot_idx, node, 0] = community_id

# Αποθήκευση του τανυστή X
torch.save(X, 'results/X_custom.pt')
print("Tensor X has been saved as 'results/X_custom.pt'")
X = X.type(torch.FloatTensor)

# -----------------------------------------------------------------------------------#
# Δημιουργία του τανυστή y_true
y_true = torch.zeros(num_snapshots, dtype=torch.float32)

# Ανάγνωση του αρχείου events.txt για να ενημερώσουμε το y_true
events_file = os.path.join(folder_path, 'events.txt')

# Σύνολο για την αποθήκευση των διαθέσιμων snapshots
valid_snapshots = set(range(1, len(graph_files) + 1))

with open(events_file, 'r', encoding='utf-8') as f:
    current_time_step = None
    for line in f:
        parts = line.strip().split(':')

        # Αν η γραμμή έχει αριθμό πριν από το ':', ενημερώνουμε το current_time_step
        if len(parts) > 1 and parts[0].strip().isdigit():
            current_time_step = int(parts[0].strip())

        # Έλεγχος αν η γραμμή περιέχει "MERGE" ή "SPLIT"
        if 'MERGE' in line or 'SPLIT' in line:
            if current_time_step in valid_snapshots:
                # Βάζουμε current_time_step - 1 με βάση το events.txt
                if 0 <= current_time_step - 1 < len(y_true):
                    y_true[current_time_step - 1] = 1

# Αποθήκευση του τανυστή y_true
torch.save(y_true, 'results/y_true_custom.pt')
print(f"The tensor y_true has been saved as 'y_true_custom.pt' in the 'results' folder.")
y_true = y_true.type(torch.FloatTensor)

