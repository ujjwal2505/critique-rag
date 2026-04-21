# Northwind Analytics — Authentication and SSO

Northwind Analytics supports two ways for team members to sign in.

## Email and password
Available on all plans. Passwords must be at least 12 characters. Two-factor
authentication via an authenticator app (TOTP) is available on all plans and is
required for all Enterprise accounts.

## Single sign-on (SSO)
SSO is available on the Pro and Enterprise plans only. Northwind supports SSO
through the SAML 2.0 protocol and integrates with the following identity
providers:

- Okta
- Microsoft Entra ID (formerly Azure Active Directory)
- Google Workspace

SCIM user provisioning (automatic creation and deactivation of accounts) is
available on the Enterprise plan only, and works with Okta and Microsoft Entra
ID. Google Workspace is supported for SSO sign-in but not for SCIM provisioning.

API access uses project-scoped API keys, which are independent of the SSO
configuration and are managed under Project Settings.
