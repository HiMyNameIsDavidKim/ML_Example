
warmup_epochs = 5
lr = 5
for epoch in range(20):
    if epoch in range(warmup_epochs):
        lr_warmup = ((epoch+1)/warmup_epochs) * lr
    print(lr_warmup)


for epoch in range(total_epochs):
    model.train()
    if epoch in range(warmup_epochs):
        lr_warmup = ((epoch+1)/warmup_epochs) * lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_warmup
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{total_epochs}] Iteration [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item()}')


