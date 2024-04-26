#%%
def train(model, data_loader, optimizer, criterion, device):

    model.train()

    total_loss = []

    for input, y in data_loader:
        

        input = input.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(input)
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        total_loss.append(loss)

    return sum(total_loss) / len(total_loss)