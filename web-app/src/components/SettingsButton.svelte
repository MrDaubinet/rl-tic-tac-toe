<!-- SettingsButton.svelte -->
<script lang="ts">
    import GiSettingsKnobs from 'svelte-icons/gi/GiSettingsKnobs.svelte'
    export let onClick = () => {};
</script>
  
  <button
    type="button"
    class="p-2 rounded-full hover:bg-gray-700 focus:outline-none"
    aria-label="Settings"
    on:click={onClick}
  >
    <div class="w-10 h-10 text-white">
      <GiSettingsKnobs />
    </div>
  </button>