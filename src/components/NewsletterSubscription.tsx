import { useState } from 'react';
import clsx from 'clsx';
import styles from './NewsletterSubscription.module.css';

interface NewsletterSubscriptionProps {
  variant?: 'inline' | 'card';
  title?: string;
  description?: string;
}

export default function NewsletterSubscription({
  variant = 'card',
  title = 'Subscribe to our newsletter',
  description = 'Get the latest news and articles delivered straight to your inbox.'
}: NewsletterSubscriptionProps): JSX.Element {
  const [email, setEmail] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [submitError, setSubmitError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitError('');

    try {
      // In a real application, you would send this to your newsletter service
      console.log('Subscribing email:', email);

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Reset form and show success message
      setEmail('');
      setSubmitSuccess(true);
      setSubmitError('');

      // Hide success message after 5 seconds
      setTimeout(() => {
        setSubmitSuccess(false);
      }, 5000);
    } catch (error) {
      setSubmitError('There was an error subscribing. Please try again.');
      console.error('Error subscribing:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className={clsx(
      'container',
      variant === 'card' ? 'padding-vert--lg padding-horiz--md' : '',
      variant === 'card' ? 'card' : '',
      styles.newsletterContainer
    )}>
      <div className="row">
        <div className={variant === 'inline' ? 'col' : 'col col--8 col--offset-2'}>
          <div className="text--center">
            <h2>{title}</h2>
            <p>{description}</p>
          </div>

          <form onSubmit={handleSubmit} className="margin-vert--lg">
            <div className={clsx('row', styles.subscriptionForm)}>
              <div className={clsx(variant === 'inline' ? 'col col--9' : 'col col--8 col--offset-2')}>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Enter your email address"
                  className={clsx('form-control', styles.emailInput)}
                  required
                />
              </div>
              <div className={clsx(variant === 'inline' ? 'col col--3' : 'col col--4')}>
                <button
                  type="submit"
                  className={clsx('button', 'button--primary', 'button--block', styles.subscribeButton)}
                  disabled={isSubmitting}
                >
                  {isSubmitting ? 'Subscribing...' : 'Subscribe'}
                </button>
              </div>
            </div>

            {submitSuccess && (
              <div className="alert alert--success margin-top--md">
                Thank you for subscribing! Please check your email to confirm your subscription.
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