import setuptools

with open("deepai_nlp/README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepai_nlp",
    version="0.0.1",
    author="DeepAI-Solutions",
    author_email="pencil.forever@gmail.com",
    description="Project for Vietnamese nlp",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    install_requires=[
        'python-crfsuite>=0.9.5',
        'sklearn-crfsuite>=0.3.6',
        'joblib>=0.12.5',
        'gensim>=3.5.0',
        'tensorflow>=1.8.0',
        'keras>=2.2.0'
    ],
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    package_data={'deepai_nlp': ['data/*/samples/*', 'models/pretrained*',
                                 'tokenization/*.txt']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
