#%%
def train(model, data_loader, optimizer, criterion, device):

    model.train()

    total_loss = []

    for x1, x2, x3, x4, y in data_loader:
        

        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        x4 = x4.to(device)

        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x1,x2,x3,x4)
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    
    return sum(total_loss) / len(total_loss)

