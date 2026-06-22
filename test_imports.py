import sys
sys.path.insert(0, '.')

print('--- Test import moduli src ---')
from src.data    import load_data
from src.model   import BatteryDecayModel
from src.loss    import mse_loss, make_constrained_loss, constrained_mse_loss
from src.trainer import train, train_lbfgsb, TrainConfig, TrainHistory
from src.utils   import (plot_loss_curve, plot_fitting_curve,
                          compare_optimizers, plot_dashboard,
                          print_summary_table, save_history, load_history)
print('[OK] Tutti i moduli importati')

import torch
import time

# Test modello
print('\n--- Test forward pass ---')
model = BatteryDecayModel()
x = torch.linspace(0, 1, 20)
y = model(x)
print(f'[OK] forward: input={x.shape}, output={y.shape}')
print(f'     Params: {model}')

# Test loss
print('\n--- Test loss functions ---')
y_true = torch.ones(20) * 1.5
l1 = mse_loss(y, y_true, model)
l2 = constrained_mse_loss(y, y_true, model)
print(f'[OK] mse_loss             = {l1.item():.6f}')
print(f'[OK] constrained_mse_loss = {l2.item():.6f}')

# Test data loading
print('\n--- Test data loading ---')
x_data, y_data, meta = load_data('battery_cycle_level_dataset_CLEAN_FINAL.csv')
n_samp = meta['n_samples']
n_bat  = meta['n_batteries']
x_min  = x_data.min().item()
x_max  = x_data.max().item()
y_min  = y_data.min().item()
y_max  = y_data.max().item()
print(f'[OK] {n_samp} campioni, {n_bat} batterie')
print(f'     x in [{x_min:.3f}, {x_max:.3f}], y in [{y_min:.3f}, {y_max:.3f}]')

# Test mini-training
print('\n--- Test mini-training (50 epoch GD) ---')
model2  = BatteryDecayModel()
opt     = torch.optim.SGD(model2.parameters(), lr=0.01)
loss_fn = make_constrained_loss(1e3)
config  = TrainConfig(epochs=50, batch_size=-1, log_every=10,
                      patience=200, use_projection=True)
t0   = time.perf_counter()
hist = train(model2, opt, x_data, y_data, loss_fn, config,
             optimizer_name='GD-test', verbose=False)
dt   = time.perf_counter() - t0
print(f'[OK] 50 epoch in {dt*1000:.1f}ms')
print(f'     Loss: {hist.losses[0]:.4f} -> {hist.losses[-1]:.4f}')
print(f'     Modello: {model2}')

# Test L-BFGS-B (10 iter)
print('\n--- Test L-BFGS-B (10 iter) ---')
model3 = BatteryDecayModel()
cfg3   = TrainConfig(epochs=10, use_projection=False)
bounds = [(1e-6, None)] * 3
hist3  = train_lbfgsb(model3, x_data, y_data, mse_loss, cfg3, bounds=bounds, verbose=False)
print(f'[OK] L-BFGS-B: {len(hist3.losses)} iterazioni, loss={hist3.losses[-1]:.6f}')
print(f'     Modello: {model3}')

print('\n=== TUTTI I TEST SUPERATI ===')
