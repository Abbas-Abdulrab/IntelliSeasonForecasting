import cherrypy
from requests_oauthlib import OAuth2Session
import requests
import json
import os

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# OAuth2 client details
client_id = ''
client_secret = ''
redirect_uri = 'http://localhost:8080/callback'
authorization_base_url = 'https://accounts.google.com/o/oauth2/auth'
token_url = 'https://oauth2.googleapis.com/token'

class OAuth2App:
    def __init__(self):
        pass

    @cherrypy.expose
    def index(self):
        """Step 1: User Authorization.
        Redirect the user to the OAuth provider (Google) using an URL with a few key OAuth parameters.
        """
        scope = ['https://www.googleapis.com/auth/cloud-platform']
        google = OAuth2Session(client_id, redirect_uri=redirect_uri, scope=scope)
        authorization_url, state = google.authorization_url(authorization_base_url, access_type="offline")

        # Save the state in the session
        cherrypy.session['oauth_state'] = state
        raise cherrypy.HTTPRedirect(authorization_url)

    @cherrypy.expose
    def callback(self, state=None, code=None, scope=None):
        """Step 2: Retrieving an access token.
        The user has been redirected back from the provider to your registered callback URL.
        With this redirection comes an authorization code included in the redirect URL.
        We will use that to obtain an access token.
        """
        # Retrieve the saved state from the session
        saved_state = cherrypy.session.get('oauth_state')
        if state != saved_state:
            return "State mismatch error. Possible CSRF attack."

        if code:
            google = OAuth2Session(client_id, redirect_uri=redirect_uri, state=saved_state)
            token = google.fetch_token(token_url, client_secret=client_secret, code=code)

            # Save the token in the session
            cherrypy.session['oauth_token'] = token
            return f'Token: {token}'
        else:
            return f"No authorization code found. Parameters received: state={state}, code={code}, scope={scope}"

    @cherrypy.expose
    def model(self, model_url=None):
        """Step 3: Accessing the model endpoint.
        Use the access token to retrieve results from the specified model URL.
        """
        oauth_token = cherrypy.session.get('oauth_token')
        if not oauth_token:
            return "No access token available. Please authenticate first."

        if not model_url:
            return "Please provide the model URL."

        headers = {
            'Authorization': f'Bearer {oauth_token["access_token"]}',
            'Content-Type': 'application/json'
        }
        # Assuming payload is required
        payload = json.dumps({
            # Replace with actual payload data needed for your model
            "series": [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            ],
            "frequency": [1, 1]
        })

        response = requests.post(model_url, headers=headers, data=payload)

        if response.status_code == 200:
            return f'Model response: {response.json()}'
        else:
            return f'Failed to retrieve model response. Status code: {response.status_code}, Response: {response.text}'

if __name__ == '__main__':
    cherrypy.quickstart(OAuth2App(), '/', config={
        '/': {
            'tools.sessions.on': True,
            'tools.sessions.timeout': 60
        }
    })