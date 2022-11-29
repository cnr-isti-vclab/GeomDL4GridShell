import torch
import time
from LacconianCalculus import LacconianCalculus
from models.layers.featured_mesh import FeaturedMesh
from models.networks import DisplacerNet, MultiDisplacerNet, MultiMaxDisplacerNet, MultiMeanDisplacerNet
from options.net_optimizer_options import NetOptimizerOptions
from utils import save_mesh, save_cloud, map_to_color_space, export_vector
from matplotlib import cm


class LacconianNetOptimizer:

    def __init__(self, file, lr, momentum, device, loss_type, no_knn, transform_in_features, get_loss, layer_mode):
        self.initial_mesh = FeaturedMesh(file=file, device=device)
        self.device = device
        self.loss_type = loss_type
        self.lacconian_calculus = LacconianCalculus(device=device, mesh=self.initial_mesh)
        self.start_loss = self.lacconian_calculus(self.initial_mesh, self.loss_type)
        self.lr = lr
        self.momentum = momentum
        self.no_knn = no_knn
        self.transform_in_features = transform_in_features
        self.get_loss = get_loss
        self.layer_mode = layer_mode

        # Setting 10 decimal digits tensor display.
        torch.set_printoptions(precision=10)

        # Setting randomization seed.
        torch.manual_seed(42)

        self.device = torch.device(device)

        self.make_optimizer()

    def make_optimizer(self):
        # Computing initial_mesh input features.
        self.initial_mesh.compute_mesh_input_features()

        # Initializing net model.
        if self.transform_in_features == True or 'multi' in self.layer_mode:
            mask = self.initial_mesh.feature_mask
        else:
            mask = None
        if self.layer_mode == 'dgcnn' or self.layer_mode == 'gat':
            self.model = DisplacerNet(self.no_knn, mode=self.layer_mode, in_feature_mask=mask).to(self.device)
        elif self.layer_mode == 'multi':
            self.model = MultiDisplacerNet(self.no_knn, mask).to(self.device)
        elif self.layer_mode == 'multimax':
            self.model = MultiMaxDisplacerNet(self.no_knn, mask).to(self.device)
        elif self.layer_mode == 'multimean':
            self.model = MultiMeanDisplacerNet(self.no_knn, mask).to(self.device)

        # Initializing model weights.
        # self.model.apply(self.model.weight_init)

        # Initializing loss list.
        if self.get_loss:
            self.loss_list = []
            self.structural_loss_list = []
            self.penalty_loss_list = []

        # Building optimizer.
        # self.optimizer = torch.optim.Adam([ self.model.parameters ], lr=lr)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        # Building lr decay scheduler.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=15, verbose=True)

    def optimize(self, n_iter, save, save_interval, display_interval, save_label, take_times, neighbor_list, save_prefix=''):
        # Initializing best loss.
        best_loss = torch.tensor(float('inf'), device=self.device)

        # Saving initial mesh with structural data.
        if save:
            vmax = 10 * self.start_loss
            filename = save_prefix + '[START]' + save_label + '.ply'
            quality = torch.norm(self.lacconian_calculus.vertex_deformations[:, :3], p=2, dim=1)
            save_mesh(self.initial_mesh, filename, v_quality=quality.unsqueeze(1))
            export_vector(self.initial_mesh.edges, 'edges.csv')
            export_vector(quality, save_prefix + '[START]load_' + save_label + '.csv')
            export_vector(self.lacconian_calculus.beam_energy, save_prefix + '[START]energy_' + save_label + '.csv')
            export_vector(map_to_color_space(self.lacconian_calculus.beam_energy.cpu(), vmin=0, vmax=vmax), save_prefix + '[START,RGBA]energy_' + save_label + '.csv', format='%d')

        for current_iteration in range(n_iter):
            iter_start = time.time()

            # Putting grads to None.
            self.optimizer.zero_grad(set_to_none=True)

            # Computing mesh displacements via net model.
            displacements = self.model(self.initial_mesh.input_features)
            offset = torch.zeros(self.initial_mesh.vertices.shape[0], 3, device=self.device)
            offset[self.lacconian_calculus.non_constrained_vertices, :] = displacements[self.lacconian_calculus.non_constrained_vertices]

            # Generating current iteration displaced mesh.
            iteration_mesh = self.initial_mesh.update_verts(offset)

            # Computing structural loss.
            structural_loss = self.lacconian_calculus(iteration_mesh, self.loss_type)

            # Computing boundary penalty term.
            constrained_vertices = torch.logical_not(self.lacconian_calculus.non_constrained_vertices)
            boundary_penalty = torch.mean(torch.norm(displacements[constrained_vertices], dim=1))
            if current_iteration == 0:
                penalty_scale = float(0.3 * structural_loss / boundary_penalty)

            # Summing loss components.
            loss = structural_loss + penalty_scale * boundary_penalty

            # Saving current iteration mesh if requested.
            if current_iteration % save_interval == 0:
                if save:
                    filename = save_prefix + save_label + '_' + str(current_iteration) + '.ply'
                    quality = torch.norm(self.lacconian_calculus.vertex_deformations[:, :3], p=2, dim=1)
                    save_mesh(iteration_mesh, filename, v_quality=quality.unsqueeze(1))
                    filename = save_prefix + '[RGBA]energy_' + save_label + '_' + str(current_iteration) + '.csv'
                    export_vector(map_to_color_space(self.lacconian_calculus.beam_energy.detach().cpu(), vmin=0, vmax=vmax), filename, format='%d')

            # Displaying loss if requested.
            if display_interval != -1 and current_iteration % display_interval == 0:
                print('*********** Iteration: ', current_iteration, ' Structural loss: ', structural_loss, '***********')
                self.structural_loss_list.append(float(structural_loss))
                self.loss_list.append(float(loss))
                self.penalty_loss_list.append(float(boundary_penalty))

            # Keeping data if loss is best.
            if structural_loss < best_loss:
                best_loss = structural_loss
                best_iteration = current_iteration

                if save:
                    best_mesh = iteration_mesh
                    best_displacements = torch.norm(self.lacconian_calculus.vertex_deformations[:, :3], p=2, dim=1)
                    best_energy = self.lacconian_calculus.beam_energy

            # Checking stopping criteria.
            if self.check_early_stopping(current_iteration):
                break

            # Computing gradients and updating optimizer
            back_start = time.time()
            loss.backward()
            back_end = time.time()
            self.optimizer.step()
            # self.scheduler.step(loss)

            # Deleting grad history in involved tensors.
            self.lacconian_calculus.clean_attributes()

            iter_end = time.time()

            # Displaying times if requested.
            if take_times:
                print('Iteration time: ' + str(iter_end - iter_start))
                print('Backward time: ' + str(back_end - back_start))
        
        # Saving best mesh, if mesh saving is enabled.
        if save and n_iter > 0:
            filename = save_prefix + '[BEST]' + save_label + '_' + str(best_iteration) + '.ply'
            save_mesh(best_mesh, filename, v_quality=best_displacements.unsqueeze(1))
            export_vector(best_displacements, '[BEST]load_' + save_label + str(best_iteration) + '.csv')
            export_vector(best_energy, '[BEST]energy_' + save_label + str(best_iteration) + '.csv')

        # Exporting loss vector, if requested.
        if self.get_loss:
            filename = 'structural_loss_' + save_label + '.csv'
            export_vector(torch.tensor(self.structural_loss_list), filename)
            filename = 'loss_' + save_label + '.csv'
            export_vector(torch.tensor(self.loss_list), filename)
            filename = 'penalty_loss_' + save_label + '.csv'
            export_vector(torch.tensor(self.penalty_loss_list), filename)

        # Exporting proximity clouds, if requested.
        if len(neighbor_list) != 0:
            with torch.no_grad():
                if hasattr(self.model, 'feature_transf'):
                    feature_transf = getattr(self.model, 'feature_transf')
                    out_list = [feature_transf(self.initial_mesh.input_features)]
                else:
                    out_list = [self.initial_mesh.input_features]
                for layer_idx in range(1, self.model.no_graph_layers + 1):
                    current_layer = getattr(self.model, 'layer_' + str(layer_idx))
                    if self.layer_mode == 'gat' or 'multi' in self.layer_mode:
                        if layer_idx == 1:
                            inp = out_list[0]
                        else:
                            inp = out_list[-1][0]
                        out_list.append(current_layer(inp, return_attention_weights=True))
                    elif self.layer_mode == 'dgcnn':
                        out_list.append(current_layer(out_list[-1]))

            if self.layer_mode == 'gat':
                for layer_idx in range(1, len(out_list)):
                    neighs, weights = out_list[layer_idx][1]
                    for vertex_idx in neighbor_list:
                        for head_idx in range(weights.shape[1]):
                            pos = list(range(self.no_knn*vertex_idx, self.no_knn*(vertex_idx + 1))) + [self.no_knn*len(self.initial_mesh.vertices) + vertex_idx]
                            points = neighs[0, pos]
                            attention_weights = weights[pos, head_idx]
                            colors = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
                            colormap = cm.get_cmap('jet')
                            filename = save_label + '_layer' + str(layer_idx) + '_vertex' + str(vertex_idx) + '_head' + str(head_idx) + '.ply'
                            save_cloud(self.initial_mesh.vertices[points, :], filename, color=colormap(colors.cpu()), quality=attention_weights)

    def check_early_stopping(self, current_iteration, check_width=50, relative_cond=0.1, absolute_cond=5e-3):
        if current_iteration == 0:
            self.ignore_count = 0
        if current_iteration != 0 and self.structural_loss_list[-1] > self.start_loss:
            self.ignore_count += 1
        if current_iteration > check_width + self.ignore_count:
            loss_variation = max(self.structural_loss_list[current_iteration-check_width : ]) - min(self.structural_loss_list[current_iteration-check_width : ])
            if loss_variation < absolute_cond and loss_variation < relative_cond * self.start_loss:
                return True
            else:
                return False      
        else:
            return False


if __name__ == '__main__':
    parser = NetOptimizerOptions()
    options = parser.parse()
    lo = LacconianNetOptimizer(options.path, options.lr, options.momentum, options.device, options.loss_type, options.no_knn, options.transform_in_features, options.get_loss, options.layer_mode)
    lo.optimize(options.n_iter, options.save, options.save_interval, options.display_interval, options.save_label, options.take_times, options.neighbor_list)

