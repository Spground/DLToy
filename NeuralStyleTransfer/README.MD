# Usage

1. download pre-trained vgg model from [https://pan.baidu.com/s/13AeEikQVkDWQwyYMiKrYcQ](https://pan.baidu.com/s/13AeEikQVkDWQwyYMiKrYcQ)
2. ensure you are in `NeuralStyleTransfer` dir, then type `tar -zxvf pretrained-model.tar.gz .` to extract file.
3. `pip install -r requirements.txt`
4. Prepare two images, one is Content Image, the other is Style Image, run `run.py` script, `python3 run.py -c content.jpg -s style.jpg -i 200`


# Example

![](https://github.com/Spground/DLToy/blob/master/NeuralStyleTransfer/louvre_generated.png)
