
# 🌀 AdaptiveScheduler: Loss-Aware Training Control

> A unified framework for **adaptive learning rate, momentum, and parameter freezing**,  
> built on top of standard PyTorch optimizers.

---

## 🌱 Motivation

Training deep models is tricky:  
- Fixed schedules (cosine, step, etc.) ignore what the loss is doing.  
- Many parameters stagnate with tiny gradients, but we keep updating them anyway.  
- Momentum and learning rate need very different behaviors in *stable* vs *noisy* phases.  

**AdaptiveScheduler** is a theory and a toolkit that unifies three axes of control:

1. **Past-history awareness** – smooth loss and gradient statistics with EMAs.  
2. **Selective participation (freezing)** – deactivate parameters with consistently tiny gradients (reversible).  
3. **Adaptive hyperparams** – adjust LR & momentum with blended, dynamic rules.  

Together, this yields training that is **faster, more stable, and more efficient**.

---

## ⚙️ Core Ideas

### 1. History Awareness
- Track EMA of loss and gradient norms.  
- React only to consistent signals, not noisy spikes.  

### 2. Freezing (Reversible)
- Parameters with grad-norm EMA below a threshold are marked `requires_grad=False`.  
- Frozen params can “wake up” later if gradients return.  
- Acts as a structural regularizer and cuts FLOPs.

### 3. Adaptive LR & Momentum
- **LR rules**: `trend`, `variance`, `grad_norm`, `cyclical`  
- **Momentum rules**: `relative`, `inverse`, `normalized`  
- Blending multiple rules gives smoother adaptation than relying on a single heuristic.

### 4. Dynamic Weighting
- Rule weights shift automatically:  
  - Early: variance & grad_norm dominate.  
  - Middle: trend dominates.  
  - Late: cyclical decay dominates.  

---

## 📦 Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/yourname/adaptive-scheduler.git
cd adaptive-scheduler
pip install -r requirements.txt
````

Dependencies:

* `torch >= 1.12`
* `numpy`

---

## 🚀 Usage

```python
import torch
from AdaptiveScheduler import AdaptiveScheduler

model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

sched = AdaptiveScheduler(
    optimizer, model,
    lr_rules={"trend": 0.6, "variance": 0.3, "grad_norm": 0.1},
    momentum_rules={"relative": 0.7, "inverse": 0.3},
    dynamic_weights=True,
    freeze_threshold=1e-4,
    freeze_ema_beta=0.9,
    min_steps_before_freeze=100,
)

for step, (x, y) in enumerate(loader, 1):
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

    snap = sched.step(loss)   # update LR, momentum, freezing
    if step % 50 == 0:
        print(f"[{step}] lr={snap['lr']:.6e} mu={snap['momentum']:.3f} frozen={snap['pct_frozen']}")
```

---

## 📊 Logging

`sched.log` keeps a rolling history of:

* LR & momentum values
* Rule weights (for LR & momentum)
* % of parameters frozen

You can easily plot these over time for diagnostics.

---

## 🧪 Why It Works

* **Efficiency**: frozen params → fewer updates.
* **Stability**: LR shrinks in noisy phases, grows when loss improves.
* **Generalization**: selective freezing regularizes unused capacity.
* **Plug-and-play**: works with any optimizer that has `lr` (and optionally `momentum`).

---

## 🔮 Roadmap

* [ ] TensorBoard logger hook
* [ ] Per-group adaptation (different rules for different layers)
* [ ] Integration with mixed-precision training
* [ ] Benchmarks on CIFAR, ImageNet, and Transformers

---

## 📖 Citation

If you use this idea in research, you can cite it as:

```
@misc{adaptive-scheduler-2025,
  author = {Your Name},
  title = {AdaptiveScheduler: Loss-Aware Training Control with Freezing},
  year = {2025},
  howpublished = {GitHub},
}
```

---

## 📝 License

MIT License

