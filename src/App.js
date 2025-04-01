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
  const [inputValues, setInputValues] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      const loadedModel = await tf.loadLayersModel("/model/rnn.json");
      setModel(loadedModel);
      console.log("Model loaded successfully.");
    };
    loadModel();
  }, []);

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

        const headers = rows[0];
        const values = rows[1].map((val) => parseFloat(val));

        // Filter out zero values and corresponding headers
        const filteredValues = values.filter((val) => val !== 0);

        setInputValues(filteredValues);
        console.log("Processed Input:", filteredValues);
      };
      reader.readAsText(file);
    }
  };

  const calculatePortfolio = async () => {
    if (!model || !inputValues) {
      alert("Please ensure the model is loaded and input data is prepared.");
      return;
    }

    const tensorInput = tf.tensor3d(
      [inputValues.map((v) => [v])],
      [1, inputValues.length, 1]
    );
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
              Based on the analysis of inflation, exchange rates, gold, FDI, and
              risk factors.
            </p>
          </Container>
        </section>

        <section id="input-data" className="py-5">
          <Container>
            <h2 className="text-center mb-4">Upload Input File</h2>
            <Row className="justify-content-center">
              <Col md={6} className="mb-3">
                <Form.Group>
                  <Form.Label>CSV File</Form.Label>
                  <Form.Control
                    type="file"
                    accept=".csv"
                    onChange={handleFileUpload}
                  />
                </Form.Group>
              </Col>
            </Row>
            <div className="text-center mt-4">
              <Button variant="success" onClick={calculatePortfolio}>
                Calculate Optimal Portfolio
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
                  {["CAD", "RMB", "EUR", "JPY", "GBP", "USD"].map(
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
          </Container>
        </section>

        <footer className="py-4 bg-dark text-light">
          <Container className="text-center">
            <p>
              &copy; 2025 Abdukarimov, Karimuratova, Kazikhanov. All rights
              reserved.
            </p>
          </Container>
        </footer>
      </header>
    </div>
  );
}

export default App;
