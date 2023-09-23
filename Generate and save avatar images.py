# Generate and save avatar images
def generate_and_save_images(generator, epoch, latent_dim, n_examples=25):
    os.makedirs("generated_images", exist_ok=True)
    z = torch.randn(n_examples, latent_dim).cuda()
    gen_imgs = generator(z)
    save_image(gen_imgs.data, f"generated_images/avatar_{epoch}.png", nrow=5, normalize=True)

# Load the trained generator
generator = Generator(latent_dim).cuda()
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()

# Generate and save avatar images after training
for epoch in range(epochs):
    generate_and_save_images(generator, epoch, latent_dim)

# Visualize some generated avatar images
sample_images = os.listdir('generated_images')
sample_images.sort()
fig, axs = plt.subplots(5, 5, figsize=(10, 10))
cnt = 0
for i in range(5):
    for j in range(5):
        img = plt.imread(os.path.join('generated_images', sample_images[cnt]))
        axs[i, j].imshow(img)
        axs[i, j].axis('off')
        cnt += 1
plt.show()
