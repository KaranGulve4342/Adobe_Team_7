## Team Name : **Team_7**

<a name="readme-top"></a>

<br />
<div align="center">
  <a href="https://github.com/KaranGulve4342/Adobe_Team_7">
    <img src="https://img.freepik.com/free-photo/sound-curve-ai-generated-image_268835-5056.jpg" alt="Logo" width="500" height="300">
  </a>

## <h1 align="center">***Curvetopia - Adobe GenSolve Hackathon - 2024***</h1>

  <p align="center">
    This project is a part of the Gensolve Adobe 2024 Hackathon. It takes in input the doodle of a shape and return the predicted shape.
    <br />
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## **About The Project**

- Our study focuses on the categorization and regularization of 2D curves, notably hand-drawn forms and doodles. 


- The major objective is to create algorithms capable of reliably identifying and categorizing different geometric forms within a given set of curves. The initiative stresses the capacity to work with flawed and irregular shapes, acknowledging that hand-drawn doodles sometimes vary from ideal geometric forms. 



- The classification method entails assessing the geometric features of input curves and using pattern recognition algorithms to discriminate between several form categories such as lines, circles, ellipses, rectangles, polygons, and stars. 



- Our project's focus on form categorization and regularization adds to the larger field of computer vision and image analysis, with possible applications in fields such as digital art, design tools, and educational Software

<!-- GETTING STARTED -->
## Running the App

### Prerequisites:

* **Git:** Ensure you have Git installed to clone the repository.
* **Python:** You'll need Python to manage the project's dependencies and run Streamlit.
* **Streamlit:** Install Streamlit using `pip install streamlit`.
* **Requirements:** Install the project's dependencies using `pip install -r requirements.txt`.




### Steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/KaranGulve4342/Adobe_Team_7
   cd CURVETOPIA_DA
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
* **Requirements:** In case there is an issue with the cairo library setup an environment using conda and install the cairosvg library using conda commands mentioned below.

3. create a new environment using conda
    ```bash
    conda create -n myenv python=3.12
    ```

4. activate the environment
    ```bash
    conda activate myenv
    ```

5. install the cairosvg library
    ```bash
    conda install -c conda-forge cairosvg
    ```

- **Note:** If you are using a different environment manager, you can install the cairosvg library using the following command:
    ```bash
    pip install cairosvg
    ```

6. **for regularization of the curves, run the following command:**
    ```bash
    python main.py
    ``` 

- **Note:** If you want to change the image, simply replace the csv path in the main.py file by the path of the csv you want to regularize.:
    ***line no. 5 of main.py file***

- **Note:** Once you run the main.py file, the regularized curves will be showed in a new window but 2 windows will be showed. One will be the given image and one will be the regularized image. 

7. **for Occlusion of the curves, run the following command:**
    ```bash
    python occlusion.py
    ``` 

- **Note:** If you want to change the image, simply replace the csv path in the occlusion.py file by the path of the csv you want to occlude.:
    ***line no. 141 of occlusion.py file***

- **Note:** Once you run the occlusion.py file, the output curves will be showed in a new window but 2 windows will be showed. One will be the given image and one will be the occluded image. 

8. **for symmetry of the curves, run the following Notebook:**
    ```bash
    Symmetry.ipynb
    ``` 

- **Note:** The .ipynb will generate a image file in .svg format called polylines.svg which will be the output of the symmetry of the curves. In case you are not able to find the file try running the symmetry.py file. It works in the same way. If you want to use a different image, simply replace the argv[i] in the symmetry.py file by the path of the argument image you want to give to the python flie.:
    ***line no. 183 of symmetry.py file***



## Contact

***If there are any issues or queries, feel free to contact us. Please..***

1. **Sanchit Chavan** - sanchitchavan3636@gmail.com
2. **Kartik Kunjekar** - kunjekarkartik@gmail.com
3. **Karan Gulve** - karanpg2306@gmail.com

Project Link: [https://github.com/KaranGulve4342/Adobe_Team_7](https://github.com/KaranGulve4342/Adobe_Team_7)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

