import torch, argparse, os, random, numpy as np
from model import UDECMO
from dataset import MMDataset
from metrics import clustering_acc, normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans

if __name__ == "__main__":
    ## === Step 1: Environment & Reproducibility Setup ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/data_sc_multiomics/TEA/', help='The data dir.')
    parser.add_argument('--output_dir', default='./result/data_sc_multiomics/HDUMEC/TEA/', help='The output dir.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--latent_dim', default=20, type=int, help='size of the latent space [default: 20]')
    parser.add_argument('--batch_size', default=2000, type=int, help='input batch size for training [default: 2000]')
    parser.add_argument('--epoch_num', default=[200, 100, 100], type=int, nargs=3, help='number of epochs per stage [default: 200 100 200]')
    parser.add_argument('--learning_rate', default=[5e-3, 1e-3, 1e-3], type=float, nargs=3, help='learning rate per stage [default: 5e-3 1e-3 1e-3]')
    parser.add_argument('--log_interval', default=1, type=int, help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--update_interval', default=1, type=int, help='how many epochs to wait before updating cluster centers [default: 10]')
    parser.add_argument('--tol', default=1e-3, type=float, help='tolerance for convergence [default: 1e-3]')
    parser.add_argument('--temperature', default=1.2, type=float, help='uncertainty-aware clustering_loss: w=exp(-uncertainty/temperature); larger temperature => closer to unweighted [default: 1.2]')
    # parser.add_argument('--verbose', default=0, type=int, help='Whether to do the statistics.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed) # Set random seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MMDataset(args.data_dir, concat_data=False); data = [x.clone().to(device) for x in dataset.X]; label = dataset.Y.clone().numpy()
    data_views = dataset.data_views; data_samples = dataset.data_samples; data_features = dataset.data_features; data_categories = dataset.categories
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_samples, shuffle=True)
    model = UDECMO(embed_dim=args.latent_dim, feature_dims=data_features, num_views=data_views, hidden_dims=[512, 512, 512], num_samples=data_samples, n_clusters=data_categories, alpha=1.0).to(device)
    ## === Stage 1: Uncertainty-Aware Reconstruction Pretraining ===
    print("\n=== Stage 1: Uncertainty-Aware Reconstruction Pretraining ===")
    print("Basic reconstruction training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[0])
    for epoch in range(args.epoch_num[0]):
        model.train()
        losses = []
        for batch_idx, (x, y, idx) in enumerate(dataloader):
            x = [x.to(device) for x in x]
            optimizer.zero_grad()
            loss = model.reconstruction_loss(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'Epoch {epoch} completed. Average Loss: {np.mean(losses):.4f}')
    embedding = model.forward_embedding(data).detach().cpu().numpy() # shape: [num_samples, embed_dim]
    kmeans = KMeans(n_clusters=data_categories, n_init=20, random_state=0)
    preds = kmeans.fit_predict(embedding)
    acc_val = clustering_acc(label, preds)
    nmi_val = normalized_mutual_info_score(label, preds)
    asw_val = silhouette_score(embedding, preds)
    ari_val = adjusted_rand_score(label, preds)
    print(f"Pretraining completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}")
    print("Uncertainty-aware reconstruction finetuning...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[1])
    for epoch in range(args.epoch_num[1]):
        model.train()
        losses = []
        for batch_idx, (x, y, idx) in enumerate(dataloader):
            x = [x.to(device) for x in x]
            optimizer.zero_grad()
            loss = model.uncertainty_aware_reconstruction_loss(x)
            loss.backward()
            # # Gradient clipping to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
        print(f'Epoch {epoch} completed. Average Loss: {np.mean(losses):.4f}')
    embedding = model.forward_embedding(data).detach().cpu().numpy() # shape: [num_samples, embed_dim]
    kmeans = KMeans(n_clusters=data_categories, n_init=20, random_state=0)
    preds = kmeans.fit_predict(embedding)
    acc_val = clustering_acc(label, preds)
    nmi_val = normalized_mutual_info_score(label, preds)
    # asw_val = 1 # silhouette_score(embedding, preds)
    ari_val = adjusted_rand_score(label, preds)
    print(f"Finetuning completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}")
    feature_level_uncertainties, modality_level_uncertainties, sample_level_uncertainties = model.get_hierarchical_uncertainty(data)

    ## === Stage 2: Deep Embedding Clustering by DEC (uncertainty-weighted KL) ===
    print("\n=== Stage 2: Deep Embedding Clustering ===")
    print("Cluster center initialization...")
    model.eval()
    embedding = model.forward_embedding(data).detach().cpu().numpy() # shape: [num_samples, embed_dim]
    kmeans = KMeans(n_clusters=data_categories, n_init=20, random_state=0)
    preds = kmeans.fit_predict(embedding)
    acc_val = clustering_acc(label, preds)
    nmi_val = normalized_mutual_info_score(label, preds)
    asw_val = silhouette_score(embedding, label)
    ari_val = adjusted_rand_score(label, preds)
    print(f"Cluster center initialization completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}")
    model.cluster_centers = kmeans.cluster_centers_ # shape: (n_clusters, embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[2])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9) # learning rate = 1e-3 * 0.9^(20/20) = 9e-4
    losses = []
    for epoch in range(args.epoch_num[2]):
        # Update target distribution periodically
        if epoch % args.update_interval == 0:
            model.eval()
            with torch.no_grad():
                q, embedding = model.forward_similarity_matrix_q(data)
                p = model.target_distribution(q)
            y_pred = torch.argmax(q, dim=1).cpu().numpy()
            acc_val = clustering_acc(label, y_pred)
            nmi_val = normalized_mutual_info_score(label, y_pred)
            asw_val = 1 # asw_val = silhouette_score(embedding, preds) # computation is very slow
            ari_val = adjusted_rand_score(label, y_pred)
            if epoch == 0:
                delta_label = 1.0
                y_pred_last = y_pred.copy()
                print(f'[Epoch {epoch}] ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}, Delta: {delta_label:.4f}')
            else:
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred.copy()
                print(f'[Epoch {epoch}] ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}, Delta: {delta_label:.4f}')
                if delta_label < args.tol:
                    print('Converged, stopping training...')
                    break
        # Training based on the target distribution that is updated periodically
        model.train()
        losses = []
        for batch_idx, (x, y, idx) in enumerate(dataloader):
            x = [x.to(device) for x in x]
            optimizer.zero_grad()
            sample_level_uncertainties_batch = torch.tensor(sample_level_uncertainties[idx.cpu().numpy()], dtype=torch.float32, device=device)
            loss = model.uncertainty_aware_clustering_loss(x, p[idx], sample_level_uncertainties_batch, temperature=args.temperature)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        print(f'Epoch {epoch} completed. Average Loss: {np.mean(losses):.4f}')
    print(f'Final ACC: {clustering_acc(label, y_pred):.4f}')
