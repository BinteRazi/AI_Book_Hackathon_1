import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import QuizComponent from '../components/QuizComponent/QuizComponent';
import LoadingSpinner from '../components/LoadingSpinner';

const QuizzesPage = () => {
  const [quizzes, setQuizzes] = useState([]);
  const [selectedQuiz, setSelectedQuiz] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch available quizzes from backend
  useEffect(() => {
    const fetchQuizzes = async () => {
      try {
        const response = await fetch('http://localhost:4000/api/quizzes');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setQuizzes(data);
      } catch (err) {
        setError(`Failed to load quizzes: ${err.message}`);
        console.error('Error fetching quizzes:', err);

        // Fallback to mock data if API fails
        const mockQuizzes = [
          {
            id: 'book_1_quiz.json',
            title: 'The Art of War Quiz',
            description: 'Test your knowledge of ancient Chinese military strategy'
          },
          {
            id: 'book_2_quiz.json',
            title: 'To Kill a Mockingbird Quiz',
            description: 'Test your knowledge of this classic American novel'
          },
          {
            id: 'book_3_quiz.json',
            title: '1984 Quiz',
            description: 'Test your knowledge of Orwell\'s dystopian novel'
          },
          {
            id: 'chapter_The_Art_of_AI_Introduction.txt_quiz.json',
            title: 'AI Introduction Quiz',
            description: 'Test your knowledge of artificial intelligence basics'
          },
          {
            id: 'chapter_The_Art_of_AI_Machine_Learning.txt_quiz.json',
            title: 'Machine Learning Quiz',
            description: 'Test your knowledge of machine learning concepts'
          }
        ];
        setQuizzes(mockQuizzes);
      } finally {
        setLoading(false);
      }
    };

    fetchQuizzes();
  }, []);

  // Function to fetch real quiz data from backend
  const fetchQuizFromBackend = async (quizId) => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:4000/api/quizzes/${quizId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const quizData = await response.json();
      setSelectedQuiz(quizData);
    } catch (err) {
      setError(`Failed to load quiz: ${err.message}`);
      console.error('Error fetching quiz from backend:', err);

      // Fallback to mock data if API fails
      const mockQuizData = {
        title: quizId.replace('.json', '').replace(/_/g, ' '),
        questions: [
          {
            question: "What is the main theme of this content?",
            options: [
              "Artificial Intelligence",
              "Machine Learning",
              "Deep Learning",
              "Natural Language Processing"
            ],
            correctAnswerIndex: 0
          },
          {
            question: "Which concept is most relevant?",
            options: [
              "Neural Networks",
              "Data Structures",
              "Algorithms",
              "Databases"
            ],
            correctAnswerIndex: 0
          },
          {
            question: "What is a key takeaway?",
            options: [
              "AI is transforming industries",
              "Programming is important",
              "Mathematics is fundamental",
              "Hardware is essential"
            ],
            correctAnswerIndex: 0
          }
        ]
      };
      setSelectedQuiz(mockQuizData);
    } finally {
      setLoading(false);
    }
  };

  if (loading && !selectedQuiz) {
    return (
      <Layout title="Quizzes" description="Interactive quizzes for AI Book Hackathon">
        <main>
          <div style={{ padding: '2rem' }}>
            <h1>Loading Quizzes...</h1>
            <LoadingSpinner />
          </div>
        </main>
      </Layout>
    );
  }

  return (
    <Layout title="Quizzes" description="Interactive quizzes for AI Book Hackathon">
      <main>
        <div style={{ padding: '2rem' }}>
          <h1>AI Book Hackathon Quizzes</h1>

          {error && (
            <div style={{ color: 'red', marginBottom: '1rem' }}>
              Error: {error}
            </div>
          )}

          {!selectedQuiz ? (
            <div>
              <p>Test your knowledge with our interactive quizzes based on books and chapters.</p>

              <div style={{ marginTop: '2rem' }}>
                <h2>Available Quizzes</h2>
                {quizzes.length > 0 ? (
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
                    gap: '1rem',
                    marginTop: '1rem'
                  }}>
                    {quizzes.map((quiz) => (
                      <div
                        key={quiz.id}
                        style={{
                          border: '1px solid #ddd',
                          borderRadius: '8px',
                          padding: '1rem',
                          backgroundColor: '#f9f9f9',
                          cursor: 'pointer'
                        }}
                        onClick={() => fetchQuizFromBackend(quiz.id)}
                      >
                        <h3>{quiz.title}</h3>
                        <p>{quiz.description}</p>
                        <button
                          style={{
                            backgroundColor: '#3498db',
                            color: 'white',
                            border: 'none',
                            padding: '0.5rem 1rem',
                            borderRadius: '4px',
                            cursor: 'pointer'
                          }}
                        >
                          Start Quiz
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p>No quizzes available at the moment.</p>
                )}
              </div>
            </div>
          ) : (
            <div>
              <button
                onClick={() => setSelectedQuiz(null)}
                style={{
                  backgroundColor: '#95a5a6',
                  color: 'white',
                  border: 'none',
                  padding: '0.5rem 1rem',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  marginBottom: '1rem'
                }}
              >
                ← Back to Quizzes
              </button>

              <QuizComponent quizData={selectedQuiz} />
            </div>
          )}
        </div>
      </main>
    </Layout>
  );
};

export default QuizzesPage;
