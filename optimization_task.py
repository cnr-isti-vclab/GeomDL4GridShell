from options.net_optimizer_options import NetOptimizerOptions
from optim.net_optimizer import StructuralNetOptimizer

# Computes a single task of shape optimization
if __name__ == '__main__':
    parser = NetOptimizerOptions()
    options = parser.parse()
    lo = StructuralNetOptimizer(options.path, options.lr, options.momentum, options.device, options.loss_type, options.no_knn, options.transform_in_features, options.get_loss, options.layer_mode)
    lo.optimize(options.max_iter, options.save, options.save_interval, options.display_interval, options.save_label, options.take_times, options.neighbor_list, options.save_prefix)