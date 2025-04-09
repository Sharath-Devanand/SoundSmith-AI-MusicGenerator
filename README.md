# SoundSmith - AI Sound Track Generator

## Project Background

The project presents a web application, SoundSmith - an AI-powered music generation
system based on voice and text input. The application utilizes cutting-edge machine learning
models - MusicGen and Riffusion for the generation of high-quality audio tracks. Advanced
Natural Language Processing techniques in the preprocessing are deployed to generate music
dynamically with a diverse range of themes and genres. The project also involves an analysis
of the models using the CLAP and the Frechet Audio Distance (FAD) scores. These metrics
provide a clear depiction of the merits and limitations of both models, along with future
improvement strategies such as hyperparameter tuning and increasing controllability. The
application with its future mobile version aims to aid songwriters to overcome creative blocks
through streamlining the ideation and inspiration phases of the audio production workflow.


## Insights

- Ethics Approval - An Ethics application process was undergone to proceed with conducting a focus group gathering requirements and critical challenges faced by the user base.

- Focus groups were conducted to connect with industry and academic songwriters and song producers and understand the current industry standards in audio production and the leading problems faced by the artists.

- A web application is designed to represent a visual interface in accessing the machine learning algorithm and present the generated audio.

### Web-design features

- The web interface emphasizes accessibility, customization, and usability for musicians and creative users, enabling both voice and text-based interactions.

- Key Features Designed:

1. Voice Command Integration - The application supports voice input, enabling hands-free interaction for musicians who may not be near recording equipment. A simple interface with Start and Stop buttons allows users to record ideas naturally. After recording, users can review the transcribed input before submission, ensuring accurate interpretation by the model.

2. Supplementary Text Prompts - A text input field allows users to add detailed instructions such as mood, genre, or instruments. This supports better customization and helps refine the musical output if the voice command alone is not descriptive enough.

3. Duration Control Slider - A slider lets users set the desired length of the generated audio, ranging from 5 to 30 seconds, offering creative flexibility based on the user's needs.

4. Model Status Feedback -  Given the computational load of generating audio with machine learning models, the interface provides real-time feedback on processing time and estimated completion, enhancing user transparency and reducing confusion during longer waits.


- The design bridges the gap between cutting-edge AI capabilities and practical user needs in the creative industry. It ensures usability for both tech-savvy and non-technical users, enhancing adoption in music ideation workflows.


### CLAP Score

- CLAP (Contrastive Language-Audio Pretraining)score is designed to assess the alignment between text and audio using text and audio embeddings.

- MusicGen achieved a higher average CLAP score than Riffusion, indicating stronger alignment with the textual input—largely due to its self-attention architecture.

- However, Riffusion showed significantly lower variance in CLAP scores, suggesting more consistent performance and reduced deviation from the intended meaning.

- Sample-wise visualizations revealed that MusicGen occasionally exceeded baseline performance but was less stable. Riffusion, on the other hand, demonstrated a coherent upward trend in CLAP score as the baseline CLAP increased—indicating more structured generalization over input quality.


- MusicGen may be preferred in applications requiring higher creativity and text alignment, such as adaptive music generation in storytelling.

- In contrast, Riffusion’s consistent performance makes it suitable for environments prioritizing predictability and stability, such as background scoring for UX/UI audio or assistive technologies.



### Frechet Audio Distance Evaluation

- evaluating the correlation between real-time audio tracks and generated audio tracks by distanced between peaks of the statistical distributions of the audio tracks

- The results indicated that Riffusion outperformed MusicGen, achieving a lower (better) FAD score. This is attributed to Riffusion's iterative generation mechanism, which progressively aligns the generated output with the structure of the baseline audio. 

- In contrast, MusicGen, while capable of producing more diverse and genre-rich outputs, often deviated from the core structure of the original audio, leading to higher FAD scores.

 - The analysis highlights Riffusion as a more reliable model for use cases where fidelity to original audio structure is critical—such as voice mimicry, sound branding, or soundtrack generation based on narrative cues.
 
 - Conversely, MusicGen may be more suitable for creative applications where diversity and musical exploration are prioritized.




