"""
email_service.py -- Transactional Email and OTP Service for Lexi (Nepal Legal AI)

Supports:
- SMTP email delivery (Gmail, Resend SMTP, SendGrid, Outlook, Mailgun, Brevo, AWS SES)
- HTML & Plaintext multipart formatting with sleek, modern 'Lexi' branding
- Personalized greeting ("Hello {Name},")
- Development / Local fallback mode: logs/prints OTP code clearly if SMTP credentials
  are not yet configured in .env so development is never blocked.
"""

from __future__ import annotations

import email.message
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "").strip() or SMTP_USER or "noreply@okf-nepal.org"
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Lexi").strip()
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() in ("true", "1", "yes")


def is_smtp_configured() -> bool:
    """Check if SMTP credentials are fully provided in .env."""
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)


def send_email(
    to_email: str,
    subject: str,
    text_content: str,
    html_content: str | None = None,
) -> bool:
    """Send an email using configured SMTP, or fallback to dev console display."""
    if not is_smtp_configured():
        # Fallback for dev / offline mode
        print("\n" + "=" * 65)
        print(f"[EMAIL SERVICE - DEV MODE] Email to: {to_email}")
        print(f"Subject: {subject}")
        print("-" * 65)
        print(text_content.strip())
        print("=" * 65)
        print("[NOTE] To send real emails, set SMTP_HOST, SMTP_USER, SMTP_PASSWORD in .env")
        print("=" * 65 + "\n")
        return True

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
    msg["To"] = to_email

    msg.attach(MIMEText(text_content, "plain", "utf-8"))
    if html_content:
        msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        if SMTP_USE_SSL or SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15) as server:
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
        return True
    except Exception as e:
        # Fallback to dev output so user is not stuck if SMTP fails
        print(f"\n[WARNING] Failed to send email via SMTP ({e}). Displaying locally:")
        print("\n" + "=" * 65)
        print(f"[FALLBACK EMAIL] To: {to_email}")
        print(f"Subject: {subject}")
        print(text_content.strip())
        print("=" * 65 + "\n")
        return False


def send_otp_email(
    to_email: str,
    otp_code: str,
    user_name: str | None = None,
    purpose: str = "signup",
    valid_minutes: int = 10,
) -> bool:
    """Format and send an OTP verification email branded with Lexi and personalized greeting."""
    action_text = "verify your email and activate your account" if purpose == "signup" else "sign in to your Lexi account"
    subject = f"[{otp_code}] Your Lexi Verification Code"

    # Personalized greeting
    clean_name = (user_name or "").strip()
    greeting = f"Hello {clean_name}," if clean_name else "Hello,"

    text_content = f"""
{greeting}

Thank you for choosing Lexi. Your verification code to {action_text} is:

    {otp_code}

This code will expire in {valid_minutes} minutes.
If you did not request this verification code, please disregard this email.

Best regards,
Lexi | Legal Intelligence Platform
Open Knowledge Foundation Nepal
"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Your Lexi Verification Code</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
      color: #1e293b;
      background-color: #f1f5f9;
      margin: 0;
      padding: 0;
      -webkit-font-smoothing: antialiased;
    }}
    .wrapper {{
      width: 100%;
      table-layout: fixed;
      background-color: #f1f5f9;
      padding: 40px 16px;
    }}
    .card {{
      max-width: 520px;
      margin: 0 auto;
      background: #ffffff;
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.08), 0 8px 10px -6px rgba(15, 23, 42, 0.04);
      border: 1px solid #e2e8f0;
    }}
    .banner {{
      background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 60%, #312e81 100%);
      padding: 34px 28px 28px;
      text-align: center;
      color: #ffffff;
    }}
    .brand-title {{
      margin: 0;
      font-size: 32px;
      font-weight: 800;
      letter-spacing: -0.5px;
      color: #ffffff;
    }}
    .brand-subtitle {{
      margin-top: 4px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 1.8px;
      text-transform: uppercase;
      color: #a5b4fc;
    }}
    .body-content {{
      padding: 32px 30px;
    }}
    .greeting {{
      font-size: 18px;
      font-weight: 600;
      color: #0f172a;
      margin-top: 0;
      margin-bottom: 12px;
    }}
    .lead-text {{
      font-size: 14px;
      color: #475569;
      line-height: 1.6;
      margin-bottom: 24px;
    }}
    .otp-card {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      text-align: center;
      padding: 24px 16px;
      margin: 20px 0 24px;
    }}
    .otp-digits {{
      font-family: 'SF Mono', Monaco, Menlo, Consolas, monospace;
      font-size: 38px;
      font-weight: 800;
      letter-spacing: 10px;
      color: #312e81;
      margin: 0;
      line-height: 1;
      padding-left: 10px; /* offset letter-spacing */
    }}
    .expiry-badge {{
      display: inline-block;
      margin-top: 12px;
      font-size: 12px;
      color: #64748b;
      background: #e2e8f0;
      padding: 4px 10px;
      border-radius: 20px;
      font-weight: 500;
    }}
    .notice-box {{
      background: #f8fafc;
      border-left: 3px solid #6366f1;
      padding: 12px 16px;
      border-radius: 4px;
      margin-top: 24px;
    }}
    .notice-box p {{
      margin: 0;
      font-size: 12px;
      color: #64748b;
      line-height: 1.5;
    }}
    .notice-box strong {{
      color: #334155;
    }}
    .footer {{
      background: #fafafa;
      padding: 20px;
      text-align: center;
      font-size: 12px;
      color: #94a3b8;
      border-top: 1px solid #f1f5f9;
    }}
    .footer a {{
      color: #6366f1;
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="card">
      <div class="banner">
        <h1 class="brand-title">Lexi</h1>
        <div class="brand-subtitle">Legal Intelligence Platform</div>
      </div>
      <div class="body-content">
        <p class="greeting">{greeting}</p>
        <p class="lead-text">
          Thank you for joining <strong>Lexi</strong>. Use the verification code below to {action_text}:
        </p>

        <div class="otp-card">
          <div class="otp-digits">{otp_code}</div>
          <div class="expiry-badge">&#x23F1; Valid for {valid_minutes} minutes</div>
        </div>

        <div class="notice-box">
          <p>
            <strong>Security Notice:</strong> Never share this code with anyone. Lexi support staff will never ask for your verification code.
          </p>
        </div>
      </div>
      <div class="footer">
        &copy; 2026 Lexi &bull; Open Knowledge Foundation Nepal
      </div>
    </div>
  </div>
</body>
</html>
"""

    return send_email(to_email, subject, text_content, html_content)
