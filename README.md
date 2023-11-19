# Transformers-from-Scratch

<p align="center">
    <img src="assets/logo.png" alt="Project logo">
</p>

## Table of Contents

- [Transformers-from-Scratch](#Transformers-from-Scratch)
  - [Table of Contents](#table-of-contents)
  - [About The Project](#about)
  - [Demo](#demo)
  - [Documentation](#documentation)
  - [File Structure](#file-structure)
  - [Contributors](#contributors)
  - [References](#references)
  - [License](#license)
  

## About

PyTorch-based custom Transformer model from scratch for English-to-Spanish translation, inspired by the "Attention is All You Need" paper. 

<p align="center">
    <img src="./assets/Transformer-architecture.png" alt="Transformer Architecture">
</p>

## Demo

## Documentation

Please refer ```/documentation``` or click <a href="https://github.com/PritK99/Transformers-from-Scratch/tree/main/documentation">here</a> for complete documentation of the project

## File Structure
```
👨‍💻Transformers-from-Scratch
 ┣ 📂assets                            // Contains all the reference gifs, images
 ┣ 📂documentation                     // Contains documentation and my notes on transformers
 ┃ ┣ 📄README.md
 ┣ 📄model.py                          // Transformer Architecture
 ┣ 📄train.py                          // Training loop
 ┣ 📄dataset.py                        // Loading & Preprocessing Dataset  
 ┣ 📄config.py 
 ┣ 📂visualization                     // Contains other visualizations
 ┃ ┣ 📄embedding.py
 ┃ ┣ 📄README.md
 ┣ 📄README.md
``` 

## Contributors

* <a href="https://github.com/PritK99">Prit Kanadiya</a>

## References
* <a href="https://www.youtube.com/watch?v=ISNdQcPhsts&t=2729s">YouTube Video</a> by Umar Jamil on developing transformers from scratch.
* <a href="https://arxiv.org/abs/1706.03762">Link</a> to ```Attention is all you need``` paper explaining transformer architecture
* <a href="https://huggingface.co/datasets/opus_books">opus_books</a> dataset by huggingface
* Amirhossein Kazemnejad's Blog on <a href="https://kazemnejad.com/blog/transformer_architecture_positional_encoding/">Positional Encodings</a>
 
## License
[MIT License](https://opensource.org/licenses/MIT)