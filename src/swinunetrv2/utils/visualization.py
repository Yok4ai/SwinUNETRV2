import matplotlib.pyplot as plt

# Plotting train vs validation metrics
plt.figure("train", (12, 6))

# Plot 1: Epoch Average Loss
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(model.avg_train_loss_values))]
y = model.avg_train_loss_values
plt.xlabel("Epoch")
plt.plot(x, y, color="red", label="Train Loss")
plt.legend()

# Plot 2: Train Mean Dice
plt.subplot(1, 2, 2)
plt.title("Train Mean Dice")
x = [i + 1 for i in range(len(model.train_metric_values))]
y = model.train_metric_values
plt.xlabel("Epoch")
plt.plot(x, y, color="green", label="Train Dice")
plt.legend()

plt.show()

# Plotting dice metrics for different categories (TC, WT, ET)
plt.figure("train", (18, 6))

# Plot 1: Train Mean Dice TC
plt.subplot(1, 3, 1)
plt.title("Train Mean Dice TC")
x = [i + 1 for i in range(len(model.train_metric_values_tc))]
y = model.train_metric_values_tc
plt.xlabel("Epoch")
plt.plot(x, y, color="blue", label="Train TC Dice")
plt.legend()

# Plot 2: Train Mean Dice WT
plt.subplot(1, 3, 2)
plt.title("Train Mean Dice WT")
x = [i + 1 for i in range(len(model.train_metric_values_wt))]
y = model.train_metric_values_wt
plt.xlabel("Epoch")
plt.plot(x, y, color="brown", label="Train WT Dice")
plt.legend()

# Plot 3: Train Mean Dice ET
plt.subplot(1, 3, 3)
plt.title("Train Mean Dice ET")
x = [i + 1 for i in range(len(model.train_metric_values_et))]
y = model.train_metric_values_et
plt.xlabel("Epoch")
plt.plot(x, y, color="purple", label="Train ET Dice")
plt.legend()

plt.show()



plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(model.epoch_loss_values))]
y = model.epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y, color="red")
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [model.hparams.val_interval * (i + 1) for i in range(len(model.metric_values))]
y = model.metric_values
plt.xlabel("epoch")
plt.plot(x, y, color="green")
plt.show()

plt.figure("train", (18, 6))
plt.subplot(1, 3, 1)
plt.title("Val Mean Dice TC")
x = [model.hparams.val_interval * (i + 1) for i in range(len(model.metric_values_tc))]
y = model.metric_values_tc
plt.xlabel("epoch")
plt.plot(x, y, color="blue")
plt.subplot(1, 3, 2)
plt.title("Val Mean Dice WT")
x = [model.hparams.val_interval * (i + 1) for i in range(len(model.metric_values_wt))]
y = model.metric_values_wt
plt.xlabel("epoch")
plt.plot(x, y, color="brown")
plt.subplot(1, 3, 3)
plt.title("Val Mean Dice ET")
x = [model.hparams.val_interval * (i + 1) for i in range(len(model.metric_values_et))]
y = model.metric_values_et
plt.xlabel("epoch")
plt.plot(x, y, color="purple")
plt.show()