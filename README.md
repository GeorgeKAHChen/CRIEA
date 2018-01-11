# CNN and Random Walk Image Edge Algorithm
Offcial Website(Just in Chinese): http://www.criea.info<br/>
Email: KazukiAmakawa@gmail.com<br/>
<br/>
`Warning: I am not confident if this program can work on windows or not. However, I did some work on compatibility. If you find this program cannot be worked on Windows systems, please link to me as soon as possible. THANKS~`<br/>
<br/>
As you see, I have just finished the program as my homework include the most of classical algorithm in Image Procecssing Class. And, of course, I copied the most of information from that file.<br/> 
As this program not finish, I will just show you how to get the right surround.<br/>
<br/>
## Surrounding
Here, I will give you the installation in UNIX(Ubuntu, Debain, Fedora, Red-hat and MacOS)<br/>
For windows, you need python3(Pay attention, PYTHON3), open-cv, python package Pillow, matplotlib, PyWavelets. I prefer to install them by "anaconda" rather than using others(But actually I hardly ever coding on windows, espacelly coding by python. So you know what I mean.). It is not necessary but a advice that you should install "git" on your PC.<br/>
<br/>
<b>For UNIX(Ubuntu, Debain, Fedora, Red-hat and MacOS)</b><br/>
STEP 1: Install Python3<br/>
For Linux Ubuntu and Debain system<br/>
```
    sudo apt upgrade
    sudo apt update
    sudo apt install python3 python3-pip ipython3
```
For Fedora and Red-hat system<br/>
```
    sudo yum upgrade
    sudo yum update
    sudo yum install python3 python3-pip ipython3
```
For Mac OS<br/>
You should insttall brew first<br/>
```
    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
And install python3<br/>
```
    brew update
    brew upgrade
    brew install python3 python3-pip ipython3
```

STEP 2: Install python package<br/>
```
    pip3 install Pillow
    pip3 install matplotlib
    pip3 install PyWavelets
```
You should also install opev-cv, however, as the installation of open-cv is so different, I cannot give you some confident method to install it. Here I will give you a reference of open-cv installation.<br/>
For Linux<br/>
https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html<br/>
For Mac OS<br/>
https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/<br/>

STEP 3: Download the program<br/>
You can download this program with git, and I will show you how to use it.<br/>
```
    git clone https://github.com/KazukiAmakawa/CRIEA/
```
