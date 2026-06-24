from transformers import ClapModel, ClapProcessor

print("Downloading CLAP model (615MB)...")
print("This only runs once — model gets cached locally.")

ClapModel.from_pretrained("laion/clap-htsat-unfused")
ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

print("Done. CLAP is ready.")