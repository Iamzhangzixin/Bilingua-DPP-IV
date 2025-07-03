from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bilingua-dpp-iv",
    version="1.0.0",
    author="Zhixing Zhang, Qiule Yu, Chen Yang, Hao Duan, Jiajiao Fang, Jiahao Xu, Changda Gong, Weihua Li, Guixia Liu, Yun Tang",
    author_email="your.email@example.com",
    description="A bimodal deep learning framework for predicting DPP-IV inhibitory peptides",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Iamzhangzixin/Bilingua-DPP-IV",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="peptides, dpp-iv, deep learning, bioinformatics, drug discovery",
    project_urls={
        "Bug Reports": "https://github.com/Iamzhangzixin/Bilingua-DPP-IV/issues",
        "Source": "https://github.com/Iamzhangzixin/Bilingua-DPP-IV",
        "Web Server": "https://lmmd.ecust.edu.cn/bilingua/",
    },
) 