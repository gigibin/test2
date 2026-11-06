import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

n_theta, n_phi = 40, 80
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
theta, phi = np.meshgrid(theta, phi)

x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
x = x + 0.1 * np.random.randn(*x.shape)

features = np.stack(
    [theta.ravel() / np.pi, np.sin(phi.ravel()), np.cos(phi.ravel())],
    axis=1,
)
targets = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

x_tensor = torch.from_numpy(features).float()
y_tensor = torch.from_numpy(targets).float()

class FitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.net(x)

model = FitNet()
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(5000):
    pred = model(x_tensor)
    loss = loss_fn(pred, y_tensor)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if i % 500 == 0:
        print(f"iter:{i}, loss:{loss.item():.4f}")

theta_grid = np.linspace(0, np.pi, 50)
phi_grid = np.linspace(0, 2 * np.pi, 100)
theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid)
grid_features = np.stack(
    [
        theta_mesh.ravel() / np.pi,
        np.sin(phi_mesh.ravel()),
        np.cos(phi_mesh.ravel()),
    ],
    axis=1,
)

with torch.no_grad():
    pred_grid = model(torch.from_numpy(grid_features).float()).numpy()

pred_x = pred_grid[:, 0].reshape(theta_mesh.shape)
pred_y = pred_grid[:, 1].reshape(theta_mesh.shape)
pred_z = pred_grid[:, 2].reshape(theta_mesh.shape)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x.ravel(), y.ravel(), z.ravel(), s=6, alpha=0.35)
ax.plot_surface(
    pred_x, pred_y, pred_z, color="r", alpha=0.5, linewidth=0
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_box_aspect([1, 1, 1])
ax.legend(
    handles=[
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="C0",
            markersize=5,
            label="raw data",
        ),
        Patch(facecolor="r", alpha=0.5, label="nn fit"),
    ]
)
plt.show()