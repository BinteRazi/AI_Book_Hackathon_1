import type {ReactNode} from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

import { useState } from 'react';

function ContactForm() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [submitError, setSubmitError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitError('');

    const formData = new FormData(e.target as HTMLFormElement);
    const data = {
      name: formData.get('name'),
      email: formData.get('email'),
      subject: formData.get('subject'),
      message: formData.get('message'),
    };

    try {
      // In a real application, you would send this to your backend
      console.log('Form submitted:', data);

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Reset form and show success message
      (e.target as HTMLFormElement).reset();
      setSubmitSuccess(true);
      setSubmitError('');

      // Hide success message after 5 seconds
      setTimeout(() => {
        setSubmitSuccess(false);
      }, 5000);
    } catch (error) {
      setSubmitError('There was an error sending your message. Please try again.');
      console.error('Error submitting form:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className={clsx('container', styles.contactContainer)}>
      <div className="row">
        <div className="col col--6 col--offset-3">
          <form id="contact-form" onSubmit={handleSubmit}>
            <div className="margin-bottom--lg">
              <label htmlFor="name" className="form-label">Name</label>
              <input
                type="text"
                id="name"
                name="name"
                className="form-control"
                placeholder="Enter your name"
                required
              />
            </div>

            <div className="margin-bottom--lg">
              <label htmlFor="email" className="form-label">Email</label>
              <input
                type="email"
                id="email"
                name="email"
                className="form-control"
                placeholder="Enter your email"
                required
              />
            </div>

            <div className="margin-bottom--lg">
              <label htmlFor="subject" className="form-label">Subject</label>
              <input
                type="text"
                id="subject"
                name="subject"
                className="form-control"
                placeholder="Enter subject"
                required
              />
            </div>

            <div className="margin-bottom--lg">
              <label htmlFor="message" className="form-label">Message</label>
              <textarea
                id="message"
                name="message"
                rows={5}
                className="form-control"
                placeholder="Enter your message"
                required
              ></textarea>
            </div>

            <button
              type="submit"
              className="button button--primary button--lg"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Sending...' : 'Send Message'}
            </button>

            {submitSuccess && (
              <div className="alert alert--success margin-top--md">
                Thank you! Your message has been sent successfully.
              </div>
            )}

            {submitError && (
              <div className="alert alert--error margin-top--md">
                {submitError}
              </div>
            )}
          </form>
        </div>
      </div>
    </div>
  );
}

export default function Contact(): ReactNode {
  return (
    <Layout
      title="Contact Us"
      description="Get in touch with us">
      <main>
        <div className={clsx('hero hero--primary', styles.heroBanner)}>
          <div className="container">
            <Heading as="h1" className="hero__title">
              Contact Us
            </Heading>
            <p className="hero__subtitle">We'd love to hear from you!</p>
          </div>
        </div>

        <div className="container padding-vert--lg">
          <div className="row">
            <div className="col col--8 col--offset-2">
              <div className="margin-vert--lg">
                <h2>Send us a message</h2>
                <p>Fill out the form below and our team will get back to you as soon as possible.</p>

                <ContactForm />

                <div className="margin-vert--lg">
                  <h3>Contact Information</h3>
                  <p><strong>Email:</strong> contact@example.com</p>
                  <p><strong>Phone:</strong> +1 (555) 123-4567</p>
                  <p><strong>Address:</strong> 123 Main St, City, State 12345</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}