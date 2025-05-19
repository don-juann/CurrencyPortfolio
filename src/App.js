import "./App.css";
import "bootstrap/dist/css/bootstrap.min.css";
import { useState, useEffect } from "react";
import {
  Container,
  Row,
  Col,
  Navbar,
  Nav,
  Button,
  Form,
  Card,
} from "react-bootstrap";
import * as tf from "@tensorflow/tfjs";

function App() {
  const [results, setResults] = useState(null);
  const [inputValues, setInputValues] = useState(null);
  const [modelType, setModelType] = useState("rnn");
  const [model, setModel] = useState(null);

  const modelInfo = {
    rnn: "Recurrent Neural Network (RNN) is well-suited for sequential data analysis, capturing temporal dependencies.",
    mlp: "Multi-Layer Perceptron (MLP) is a feedforward neural network ideal for simple nonlinear pattern recognition.",
    lstm: "Long Short-Term Memory (LSTM) networks handle long-term dependencies in time series data effectively.",
    linreg:
      "Linear Regression is a simple statistical model used to model linear relationships between variables.",
    rf: "Random Forest is an ensemble learning method using multiple decision trees for robust predictions.",
    gbr: "Gradient Boosting is an advanced ensemble technique that builds strong predictive models through iterative learning.",
  };

  useEffect(() => {
    const tfModels = ["mlp", "rnn", "lstm"];
    if (tfModels.includes(modelType)) {
      const loadModel = async () => {
        try {
          const loadedModel = await tf.loadLayersModel(
            `/model/${modelType}/${modelType}.json`
          );
          setModel(loadedModel);
          console.log(`${modelType} model loaded successfully.`);
        } catch (error) {
          console.error("Model loading failed:", error);
        }
      };
      loadModel();
    } else {
      setModel(null);
    }
  }, [modelType]);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target.result;
        const rows = text
          .trim()
          .split("\n")
          .map((row) => row.split(","));
        const values = rows[1].map((val) => parseFloat(val));
        const filteredValues = values.filter((val) => val !== 0 && !isNaN(val));

        if (filteredValues.length !== 25) {
          alert("Input must contain exactly 25 numerical values.");
          return;
        }

        setInputValues(filteredValues);
        console.log("Processed Input:", filteredValues);
      };
      reader.readAsText(file);
    }
  };

  const calculatePortfolio = async () => {
    if (!inputValues) {
      alert("Please upload valid input data.");
      return;
    }

    const backendModels = ["linreg", "rf", "gbr"];

    if (backendModels.includes(modelType)) {
      try {
        const response = await fetch("http://localhost:5000/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ input: inputValues, model: modelType }),
        });

        const data = await response.json();

        if (data.portfolio) {
          setResults(data.portfolio);
        } else {
          alert("Prediction failed: " + (data.error || "Unknown error"));
        }
      } catch (err) {
        console.error(err);
        alert("Failed to connect to backend.");
      }
    } else {
      if (!model) {
        alert("Model not loaded yet.");
        return;
      }

      let tensorInput;
      if (["rnn", "lstm"].includes(modelType)) {
        tensorInput = tf.tensor3d(
          [inputValues.map((v) => [v])],
          [1, inputValues.length, 1]
        );
      } else {
        tensorInput = tf.tensor2d([inputValues], [1, inputValues.length]);
      }
      const predictions = await model.predict(tensorInput).array();
      const normalizedResults = predictions[0].map(
        (value) => (value / predictions[0].reduce((a, b) => a + b, 0)) * 100
      );

      setResults(normalizedResults);
    }
  };

  return (
    <div className="App light-theme">
      <section
        className="py-5 text-center"
        style={{ backgroundColor: "#A5D5E6", color: "#1D242B" }}
      >
        <Container>
          <h1 className="display-3">Optimal Currency Portfolio Builder</h1>
          <p className="lead">
            Powered by Machine Learning models (MLP, RNN, LSTM, Random Forest,
            and more).
          </p>
        </Container>
      </section>

      <section
        id="input-data"
        className="py-5"
        style={{ backgroundColor: "#FAFAFA" }}
      >
        <Container>
          <Row>
            <Col md={7}>
              <h2 className="text-center mb-4" style={{ color: "#0077C0" }}>
                Upload Input File
              </h2>
              <Row className="justify-content-center">
                <Col md={12}>
                  <Form.Group className="mb-3">
                    <Form.Label style={{ color: "#0077C0" }}>
                      Choose Model
                    </Form.Label>
                    <Form.Select
                      value={modelType}
                      onChange={(e) => setModelType(e.target.value)}
                    >
                      <option value="rnn">Recurrent Neural Network</option>
                      <option value="mlp">Multi-Layer Perceptron</option>
                      <option value="lstm">Long Short-Term Memory</option>
                      <option value="linreg">Linear Regression</option>
                      <option value="rf">Random Forest</option>
                      <option value="gbr">Gradient Boosting</option>
                    </Form.Select>
                  </Form.Group>

                  <Card
                    className="my-4 shadow-lg"
                    style={{ backgroundColor: "#C7EEFF" }}
                  >
                    <Card.Body>
                      <Card.Title style={{ color: "#0077C0" }}>
                        About Model
                      </Card.Title>
                      <Card.Text style={{ color: "#0077C0" }}>
                        {modelInfo[modelType]}
                      </Card.Text>
                    </Card.Body>
                  </Card>

                  <Form.Group>
                    <Form.Label style={{ color: "#0077C0" }}>
                      Upload CSV
                    </Form.Label>
                    <Form.Control
                      type="file"
                      accept=".csv"
                      onChange={handleFileUpload}
                    />
                  </Form.Group>

                  <div className="text-center mt-4">
                    <Button
                      variant="primary"
                      className="btn-lg shadow"
                      style={{
                        backgroundColor: "#0077C0",
                        borderColor: "#0077C0",
                      }}
                      onClick={calculatePortfolio}
                    >
                      Calculate Optimal Portfolio
                    </Button>
                  </div>
                </Col>
              </Row>
            </Col>

            <Col md={5}>
              <h2 className="text-center mb-4" style={{ color: "#0077C0" }}>
                Portfolio Results
              </h2>
              <div className="text-center">
                {results ? (
                  <ul className="list-unstyled">
                    {["EUR", "RMB", "USD", "GBP", "CAD", "JPY"].map(
                      (currency, index) => (
                        <li key={index}>
                          <img
                            src={`/flags/${currency.toLowerCase()}.png`}
                            alt={currency}
                            style={{ width: "20px", marginRight: "10px" }}
                          />
                          {currency} - {results[index].toFixed(2)}%
                        </li>
                      )
                    )}
                  </ul>
                ) : (
                  <p className="text-muted">
                    Results will be displayed here after calculation.
                  </p>
                )}
              </div>
            </Col>
          </Row>
        </Container>
      </section>

      <footer
        className="py-4"
        style={{ backgroundColor: "#1D242B", color: "#D1D1D1" }}
      >
        <Container className="text-center">
          <p className="text-white">
            A. Abdukarimov, A. Karimuratova, Z. Kazikhanov. All rights reserved.
            &copy; 2025
          </p>
        </Container>
      </footer>
    </div>
  );
}

export default App;
