{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :variables:
   :module-variables:

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Module Variables

   .. autosummary::
      {% for item in attributes %}
      {{ item }}
      {%- endfor %}

   {% for item in attributes %}
   .. autodata:: {{ item }}
      :annotation:
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block modules %}
   {% if modules %}
   .. rubric:: Submodules

   .. autosummary::
      :toctree:
      :recursive:

      {% for item in modules %}
      {{ item }}
      {%- endfor %}
   {% endif %}
   {% endblock %}
