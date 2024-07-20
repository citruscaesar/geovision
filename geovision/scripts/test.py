# Simple training loops with explicit steps for sanity checking various operations applied during training or evaluation 
# Also add profiling support for debugging memory or speed problems # from torchvision.models import resnet18, ResNet18_Weights
# datamodule.setup("fit")# val_dl = datamodule.val_dataloader()# train_dl = datamodule.train_dataloader()
# model = resnet18(num_classes = 10)# model.fc = torch.nn.Linear(512, 10, True)# model.fc.weight = torch.nn.init.kaiming_normal_(model.fc.weight)# model_path = "resnet18_pre_imagenette.pt"# torch.save(model, model_path)
# train_loss = list() # val_loss = list()
# model = torch.load(model_path)# criterion = torch.nn.CrossEntropyLoss()# optimizer = torch.optim.Adam(model.parameters(), lr = 0.01) 
# model.to("cuda")# for epoch in range(1):
# # model.train()
# # for batch in tqdm(train_dl, desc = "train:"):
    # # images, labels = batch[0].to("cuda"), batch[1].to("cuda")
    # # preds = model(images)
    # # loss = criterion(preds, labels)
    # # train_loss.append(loss.item())
    # # loss.backward()
    # # optimizer.step()
    # # optimizer.zero_grad()
    # # del images, labels

# model.eval()
# for batch in tqdm(val_dl, desc = "val: "):
    # images, labels = batch[0].to("cuda"), batch[1].to("cuda")
    # preds = model(images)
    # loss = criterion(preds, labels)
    # val_loss.append(loss.item())
    # del images, labels
# torch.save(model, model_path)
# import matplotlib.pyplot as plt
# tlx, tly = list(range(0, len(train_loss))), train_loss
# vlx, vly = list(range(0, len(val_loss))), val_loss
# fig, (t_ax, v_ax) = plt.subplots(1, 2, figsize = (10, 5))
# t_ax.plot(tlx, tly)
# v_ax.plot(vlx, vly)