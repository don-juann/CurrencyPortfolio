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
} from "react-bootstrap";
import * as tf from "@tensorflow/tfjs";

function App() {
  const [model, setModel] = useState(null);
  const [results, setResults] = useState(null);
  const [fileData, setFileData] = useState({
    inflation: null,
    exchange: null,
    gold: null,
  });
  const [inputValues, setInputValues] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      const loadedModel = await tf.loadLayersModel("/model/model.json");
      setModel(loadedModel);
      console.log("Model loaded successfully.");
    };
    loadModel();
  }, []);

  const handleFileUpload = (e, fileType) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target.result;
        const rows = text
          .trim()
          .split("\n")
          .map((row) => row.split(","));
        setFileData((prev) => ({
          ...prev,
          [fileType]: rows,
        }));
      };
      reader.readAsText(file);
    }
  };

  const processFiles = () => {
    if (!fileData.inflation || !fileData.exchange || !fileData.gold) {
      alert("Please upload all three files.");
      return;
    }

    try {
      const inflationValues = fileData.inflation[1].map((val) =>
        parseFloat(val)
      );
      const exchangeValues = fileData.exchange[1].map((val) => parseFloat(val));
      const goldValue = parseFloat(fileData.gold[1][0]);

      const combinedInput = [...inflationValues, ...exchangeValues, goldValue];
      setInputValues(combinedInput);
      console.log("Processed Input:", combinedInput);
    } catch (error) {
      alert("Error processing files. Please check the file format.");
    }
  };

  const calculatePortfolio = async () => {
    if (!model || !inputValues) {
      alert("Please ensure the model is loaded and input data is prepared.");
      return;
    }

    const tensorInput = tf.tensor2d([inputValues], [1, 11]); // Single row with 11 features
    const predictions = await model.predict(tensorInput).array();

    // Normalize predictions to sum to 100
    const normalizedResults = predictions[0].map(
      (value) => (value / predictions[0].reduce((a, b) => a + b, 0)) * 100
    );

    setResults(normalizedResults);
  };

  return (
    <div className="App">
      <header className="banner">
        <Navbar bg="light" expand="lg" className="border-bottom">
          <Container>
            <Navbar.Brand href="#">
              Optimal Currency Portfolio Builder
            </Navbar.Brand>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
              <Nav className="ms-auto">
                <Nav.Link href="#input-data">Enter Data</Nav.Link>
                <Nav.Link href="#results">Results</Nav.Link>
              </Nav>
            </Navbar.Collapse>
          </Container>
        </Navbar>

        <section className="py-5 text-center bg-light">
          <Container>
            <h1>Optimal Currency Portfolio Builder</h1>
            <p className="lead">
              Based on the analysis of inflation, exchange rates, and gold
              value.
            </p>
            <Button variant="success" size="lg" href="#input-data">
              Start
            </Button>
          </Container>
        </section>

        <section id="input-data" className="py-5">
          <Container>
            <h2 className="text-center mb-4">Upload Input Files</h2>
            <Row>
              <Col md={4} className="mb-3">
                <Form.Group>
                  <Form.Label>Inflation CSV</Form.Label>
                  <Form.Control
                    type="file"
                    accept=".csv"
                    onChange={(e) => handleFileUpload(e, "inflation")}
                  />
                </Form.Group>
              </Col>
              <Col md={4} className="mb-3">
                <Form.Group>
                  <Form.Label>Exchange Rates CSV</Form.Label>
                  <Form.Control
                    type="file"
                    accept=".csv"
                    onChange={(e) => handleFileUpload(e, "exchange")}
                  />
                </Form.Group>
              </Col>
              <Col md={4} className="mb-3">
                <Form.Group>
                  <Form.Label>Gold Price CSV</Form.Label>
                  <Form.Control
                    type="file"
                    accept=".csv"
                    onChange={(e) => handleFileUpload(e, "gold")}
                  />
                </Form.Group>
              </Col>
            </Row>
            <div className="text-center mt-4">
              <Button variant="success" onClick={processFiles}>
                Process Files
              </Button>
            </div>
          </Container>
        </section>

        <section id="results" className="py-5 bg-light">
          <Container>
            <h2 className="text-center mb-4">Portfolio Results</h2>
            <div className="text-center">
              {results ? (
                <ul className="list-unstyled">
                  {["AUD", "EUR", "GBP", "JPY", "USD"].map(
                    (currency, index) => (
                      <li key={index}>
                        {currency}: {results[index].toFixed(2)}%
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
            <div className="text-center mt-4">
              <Button variant="success" onClick={calculatePortfolio}>
                Calculate Optimal Portfolio
              </Button>
            </div>
          </Container>
        </section>

        <footer className="py-4 bg-dark text-light">
          <Container className="text-center">
            <p>&copy; 2025 Team AI. All rights reserved.</p>
          </Container>
        </footer>
      </header>
    </div>
  );
}

export default App;
