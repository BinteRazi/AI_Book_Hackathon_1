import React, { useState } from 'react';
import styles from './QuizComponent.module.css';

const QuizComponent = ({ quizData }) => {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [showResult, setShowResult] = useState(false);
  const [score, setScore] = useState(0);

  const handleAnswer = (answerIndex) => {
    if (showResult) return;

    const newSelectedAnswers = {
      ...selectedAnswers,
      [currentQuestion]: answerIndex
    };
    setSelectedAnswers(newSelectedAnswers);
  };

  const handleNext = () => {
    if (currentQuestion < quizData.questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    }
  };

  const handlePrevious = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1);
    }
  };

  const handleSubmit = () => {
    let newScore = 0;
    quizData.questions.forEach((question, index) => {
      if (selectedAnswers[index] === question.correctAnswerIndex) {
        newScore++;
      }
    });
    setScore(newScore);
    setShowResult(true);
  };

  const handleReset = () => {
    setCurrentQuestion(0);
    setSelectedAnswers({});
    setShowResult(false);
    setScore(0);
  };

  if (quizData.questions.length === 0) {
    return <div className={styles.quizContainer}>No quiz questions available.</div>;
  }

  const currentQ = quizData.questions[currentQuestion];
  const selectedAnswerIndex = selectedAnswers[currentQuestion];

  return (
    <div className={styles.quizContainer}>
      <h3 className={styles.quizTitle}>{quizData.title || 'Quiz'}</h3>

      {!showResult ? (
        <div className={styles.questionSection}>
          <div className={styles.questionCounter}>
            Question {currentQuestion + 1} of {quizData.questions.length}
          </div>

          <h4 className={styles.questionText}>{currentQ.question}</h4>

          <div className={styles.answersContainer}>
            {currentQ.options.map((option, index) => (
              <div
                key={index}
                className={`${styles.answerOption} ${
                  selectedAnswerIndex === index ? styles.selectedAnswer : ''
                } ${showResult && index === currentQ.correctAnswerIndex ? styles.correctAnswer : ''}
                ${showResult && selectedAnswerIndex === index && index !== currentQ.correctAnswerIndex ? styles.incorrectAnswer : ''}`}
                onClick={() => handleAnswer(index)}
              >
                <span className={styles.optionLetter}>
                  {String.fromCharCode(65 + index)}.
                </span>
                <span className={styles.optionText}>{option}</span>
              </div>
            ))}
          </div>

          <div className={styles.navigationButtons}>
            <button
              onClick={handlePrevious}
              disabled={currentQuestion === 0}
              className={styles.navButton}
            >
              Previous
            </button>

            {currentQuestion < quizData.questions.length - 1 ? (
              <button
                onClick={handleNext}
                className={styles.navButton}
              >
                Next
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                className={styles.submitButton}
              >
                Submit Quiz
              </button>
            )}
          </div>
        </div>
      ) : (
        <div className={styles.resultSection}>
          <h4 className={styles.resultTitle}>Quiz Results</h4>
          <p className={styles.score}>
            You scored <strong>{score}</strong> out of <strong>{quizData.questions.length}</strong>
          </p>
          <p className={styles.percentage}>
            Percentage: <strong>{Math.round((score / quizData.questions.length) * 100)}%</strong>
          </p>

          <div className={styles.detailedResults}>
            <h5>Detailed Results:</h5>
            {quizData.questions.map((question, index) => {
              const userAnswer = selectedAnswers[index];
              const isCorrect = userAnswer === question.correctAnswerIndex;

              return (
                <div key={index} className={styles.questionResult}>
                  <p className={styles.resultQuestion}>{index + 1}. {question.question}</p>
                  <p className={styles.resultAnswer}>
                    Your answer: <span className={isCorrect ? styles.correctText : styles.incorrectText}>
                      {userAnswer !== undefined ? `${String.fromCharCode(65 + userAnswer)}. ${question.options[userAnswer]}` : 'Not answered'}
                    </span>
                  </p>
                  {!isCorrect && userAnswer !== question.correctAnswerIndex && (
                    <p className={styles.correctAnswerText}>
                      Correct answer: {String.fromCharCode(65 + question.correctAnswerIndex)}. {question.options[question.correctAnswerIndex]}
                    </p>
                  )}
                </div>
              );
            })}
          </div>

          <button
            onClick={handleReset}
            className={styles.resetButton}
          >
            Retake Quiz
          </button>
        </div>
      )}
    </div>
  );
};

export default QuizComponent;