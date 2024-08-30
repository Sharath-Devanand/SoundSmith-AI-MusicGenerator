from audiocraft.models import MusicGen
import os
import torchaudio

prompts_genres = [
    "Create a powerful orchestral composition inspired by Beethoven's symphonies, focusing on dramatic strings, woodwinds, and brass.",
    "Generate a high-energy, bass-heavy EDM track with pulsating beats, synths, and drops designed for a festival crowd.",
    "Compose a mellow, chill lo-fi hip-hop track with soft drums, vinyl crackle, jazzy chords, and relaxing vibes.",
    "Design an uplifting indie rock song featuring jangly guitars, heartfelt vocals, and catchy rhythms in the style of early 2000s bands.",
    "Create a smooth jazz-fusion track with complex improvisation on saxophone, electric guitar, and keyboards, blending jazz and rock elements.",
    "Generate an ethereal ambient soundscape filled with subtle drones, soft pads, and calming nature sounds, perfect for relaxation or meditation.",
    "Compose a laid-back reggae track with offbeat guitar skanks, deep basslines, and a groove that embodies the spirit of the Caribbean.",
    "Create a warm and heartfelt country folk song with acoustic guitar, banjo, harmonica, and soulful storytelling lyrics.",
    "Produce an upbeat Afrobeat track featuring rhythmic percussion, brass horns, and vibrant melodies inspired by African dance music.",
    "Design an intense and aggressive heavy metal track with fast-paced guitar riffs, double bass drumming, and raw, powerful vocals."
]

mood_prompts = [
    "Compose a melancholic piano ballad that evokes a deep sense of sadness and reflection, with slow, emotional melodies.",
    "Generate a joyful and upbeat pop song filled with catchy hooks, bright chords, and an infectious feel-good energy.",
    "Create a haunting, eerie soundscape featuring dissonant tones, whispering winds, and unsettling effects to evoke a sense of fear and tension.",
    "Design a calming and peaceful acoustic track with soft guitar strumming and gentle melodies, perfect for relaxation or unwinding after a long day.",
    "Produce a high-energy, adrenaline-pumping rock anthem with intense guitar riffs, driving drums, and powerful vocals, perfect for an action-packed atmosphere.",
    "Generate a dreamy, nostalgic synthwave track that evokes the bittersweet feeling of longing for the past, with lush synths and retro beats.",
    "Compose an intense, dramatic orchestral piece that builds tension and suspense, featuring pounding percussion and rising string sections.",
    "Create a romantic and tender love song with smooth vocals, delicate piano, and soft strings, capturing the feeling of intimacy and warmth.",
    "Design an empowering, triumphant hip-hop track with bold beats, confident lyrics, and a sense of determination and victory.",
    "Produce a serene and ethereal ambient track with soft, floating pads and distant chimes that evoke a sense of wonder and introspection."
]


captions = []
with open('captions.txt', 'r') as f:
    content = f.read()

# Split the content based on double newlines to separate captions
sections = content.strip().split('\n\n')

for section in sections:
    # Each section starts with an index and a period, so split it
    if '. ' in section:
        _, caption = section.split('. ', 1)
        captions.append(caption.strip())

model = MusicGen.get_pretrained('facebook/musicgen-small')

model.set_generation_params(
        use_sampling=True,
        duration=5,
        top_k=250,
        temperature=0.5,
    )

for i in range(0,5):
    output = model.generate(captions[i], progress=True, return_tokens = True)
    output = output[0]
    filepath = 'musicgen_audio/'
    sample_rate = 32000
    assert output.dim() == 2 or output.dim() == 3
    output = output.detach().cpu()
    if output.dim() == 2:
        output = output[None, ...]

    for i, o in enumerate(output):
        filename = os.path.join(filepath, f'audio_{i}.wav')
        torchaudio.save(filename, o, sample_rate)

    print(f'Generated audio for caption {i+1}')



for i in range(5,10):
    output = model.generate(captions[i], progress=True, return_tokens = True)
    output = output[0]
    filepath = 'musicgen_audio/'
    sample_rate = 32000
    assert output.dim() == 2 or output.dim() == 3
    output = output.detach().cpu()
    if output.dim() == 2:
        output = output[None, ...]

    for i, o in enumerate(output):
        filename = os.path.join(filepath, f'audio_{i}.wav')
        torchaudio.save(filename, o, sample_rate)

    print(f'Generated audio for caption {i}')

    
for i in range(0,5):
    output = model.generate(prompts_genres[i], progress=True, return_tokens = True)
    output = output[0]
    filepath = 'musicgen_audio/'
    sample_rate = 32000
    assert output.dim() == 2 or output.dim() == 3
    output = output.detach().cpu()
    if output.dim() == 2:
        output = output[None, ...]

    for i, o in enumerate(output):
        filename = os.path.join(filepath, f'genre_{i}.wav')
        torchaudio.save(filename, o, sample_rate)

    print(f'Generated audio for caption {i+1}')


for i in range(5,10):
    output = model.generate(prompts_genres[i], progress=True, return_tokens = True)
    output = output[0]
    filepath = 'musicgen_audio/'
    sample_rate = 32000
    assert output.dim() == 2 or output.dim() == 3
    output = output.detach().cpu()
    if output.dim() == 2:
        output = output[None, ...]

    for i, o in enumerate(output):
        filename = os.path.join(filepath, f'genre_{i}.wav')
        torchaudio.save(filename, o, sample_rate)

    print(f'Generated audio for caption {i+1}')


for i in range(0,5):
    output = model.generate(mood_prompts[i], progress=True, return_tokens = True)
    output = output[0]
    filepath = 'musicgen_audio/'
    sample_rate = 32000
    assert output.dim() == 2 or output.dim() == 3
    output = output.detach().cpu()
    if output.dim() == 2:
        output = output[None, ...]

    for i, o in enumerate(output):
        filename = os.path.join(filepath, f'mood_{i}.wav')
        torchaudio.save(filename, o, sample_rate)

    print(f'Generated audio for caption {i+1}')


for i in range(5,10):
    output = model.generate(mood_prompts[i], progress=True, return_tokens = True)
    output = output[0]
    filepath = 'musicgen_audio/'
    sample_rate = 32000
    assert output.dim() == 2 or output.dim() == 3
    output = output.detach().cpu()
    if output.dim() == 2:
        output = output[None, ...]

    for i, o in enumerate(output):
        filename = os.path.join(filepath, f'mood_{i}.wav')
        torchaudio.save(filename, o, sample_rate)

    print(f'Generated audio for caption {i+1}')


