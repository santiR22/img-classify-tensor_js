const webcamElement = document.getElementById("webcam");
const classifier = knnClassifier.create();
let net;

async function app() {
  // Cargar el modelo.
  net = await mobilenet.load();
  console.log("Modelo cargado correctamente");

  // Crea un objeto de tensorFlow desde la cámara
  const webcam = await tf.data.webcam(webcamElement);

  // Lee la imagen y lo asocia a una clase
  const addExample = async (classId) => {
    // Captura la imagen desde la cámara.
    const img = await webcam.capture();

    // Obtener activación MobileNet
    const activation = net.infer(img, true);

    // Pasar la activación al clasificador
    classifier.addExample(activation, classId);

    // Liberar memoria
    img.dispose();
  };

  //Guardar el entrenamiento en el formato de Tensores

  const saveModel = (async) => {
    let dataset = classifier.getClassifierDataset();
    var datasetObj = {};
    Object.keys(dataset).forEach((key) => {
      let data = dataset[key].dataSync();
      datasetObj[key] = Array.from(data);
    });

    var a = document.createElement("a");
    var file = new Blob([JSON.stringify(datasetObj)], { type: "text/plain" });
    a.href = URL.createObjectURL(file);
    a.download = "entrenamiento.txt";
    a.click();
  };

  document
    .getElementById("file-selector")
    .addEventListener("change", function () {
      var fr = new FileReader();
      fr.onload = function () {
        let tensorObj = JSON.parse(fr.result);
        Object.keys(tensorObj).forEach((key) => {
          tensorObj[key] = tf.tensor(tensorObj[key], [
            tensorObj[key].length / 1024,
            1024,
          ]);
        });
        classifier.setClassifierDataset(tensorObj);
      };

      fr.readAsText(this.files[0]);
    });

  //----------------------------------------------------------------------------------

  // Se guarda un ejemplo a la clase
  document
    .getElementById("class-a")
    .addEventListener("click", () => addExample(0));
  document
    .getElementById("class-b")
    .addEventListener("click", () => addExample(1));
  document
    .getElementById("class-c")
    .addEventListener("click", () => addExample(2));
  document
    .getElementById("class-d")
    .addEventListener("click", () => addExample(3));
  document
    .getElementById("save")
    .addEventListener("click", () => saveModel());

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();

      // Activación mobilenet desde la webcam.
      const activation = net.infer(img, "conv_preds");

      // Obtiene la clase más probable y la probabilidad.
      const result = await classifier.predictClass(activation);

      const classes = ["Naranja", "Manzana", "Banana", "Limón"];
      document.getElementById("mostrarMensaje").innerHTML = `
            <h2>Fruta Reconocida: </h2>  <h4>${classes[result.label]}\n</h4> 
            <h2>Probabilidad: </h2>  <h4> ${
              result.confidences[result.label]
            }</h4>
            `;

      // Liberar Memoria
      img.dispose();
    }

    await tf.nextFrame();
  }
}

app();
